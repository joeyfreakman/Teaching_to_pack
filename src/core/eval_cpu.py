from omegaconf import DictConfig
import argparse
import math
import wandb
import torch
import numpy as np
import threading
from tqdm import tqdm
from torchvision import transforms
from einops import rearrange
import matplotlib.pyplot as plt
from scripts.data_pruning import crop_resize
import pickle
from torch.optim.lr_scheduler import LambdaLR
import signal
import cv2
import os
import hydra
from src.model.util import set_seed, detach_dict, compute_dict_mean
from src.model.enc_dataset import load_merged_data
from src.policy.ddpm import DiffusionPolicy
from src.model.util import is_multi_gpu_checkpoint, memory_monitor, memory_monitor
from src.aloha.aloha_scripts.constants import DT, PUPPET_GRIPPER_JOINT_OPEN
from src.config.dataset_config import TASK_CONFIGS, DATA_DIR
from src.aloha.aloha_scripts.real_env import make_real_env  
from src.aloha.aloha_scripts.robot_utils import move_grippers
from src.aloha.aloha_scripts.visualize_episodes import save_videos
CROP_TOP = True  # for aloha pro, whose top camera is high
CKPT = 0  # 0 for policy_last, otherwise put the ckpt number here

def signal_handler(sig, frame):
    exit()
    
def main(args):
    set_seed(42)

    signal.signal(signal.SIGINT, signal_handler)
    threading.Thread(
        target=memory_monitor, daemon=True
    ).start()  # Start the memory monitor thread

    # Command line parameters
    is_eval = args["eval"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    num_epochs = args["num_epochs"]
    log_wandb = args["log_wandb"]
    

    # Set up wandb
    if log_wandb:
        if is_eval:
            log_wandb = False
        else:
            run_name = ckpt_dir.split("/")[-1] + f".{args['seed']}"
            wandb_run_id_path = os.path.join(ckpt_dir, "wandb_run_id.txt")
            if os.path.exists(wandb_run_id_path):
                with open(wandb_run_id_path, "r") as f:
                    saved_run_id = f.read().strip()
                wandb.init(
                    project="teachingtopack",
                    entity="joeywang-of",
                    name=run_name,
                    resume=saved_run_id,
                )
            else:
                wandb.init(
                    project="teachingtopack",
                    entity="joeywang-of",
                    name=run_name,
                    config=args,
                    resume="allow",
                )
                os.makedirs(os.path.dirname(wandb_run_id_path), exist_ok=True)
                with open(wandb_run_id_path, "w") as f:
                    f.write(wandb.run.id)

    # get task parameters
    dataset_dirs = []
    num_episodes_list = []
    max_episode_len = 0

    for task in task_name:
        task_config = TASK_CONFIGS[task]
        dataset_dirs.append(DATA_DIR)
        num_episodes_list.append(task_config["num_episodes"])
        max_episode_len = max(max_episode_len, task_config["episode_len"])
        camera_names = task_config["camera_names"]

    max_skill_len = (
        args["max_skill_len"] if args["max_skill_len"] is not None else max_episode_len
    )

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5

    policy_config = {
        "lr": args["lr"],
        "camera_names": camera_names,
        "action_dim": 14,
        "observation_horizon": 3,
        "action_horizon": 8,  # TODO not used
        "prediction_horizon": args["chunk_size"],
        "num_queries": args["chunk_size"],
        "num_inference_timesteps": 10,
        "multi_gpu": args["multi_gpu"],
        "is_eval": is_eval,
    }

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": max_episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "log_wandb": log_wandb,
        "max_skill_len": max_skill_len,
    }

    if is_eval:
        print(f"{CKPT=}")
        ckpt_names = (
            [f"policy_last.ckpt"] if CKPT == 0 else [f"policy_epoch_{CKPT}_seed_0.ckpt"]
        )
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_ddpm(
                config, ckpt_name, save_episode=True, dataset_dirs=dataset_dirs
            )
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        print()
        exit()


def make_policy(policy_class, policy_config):
    if policy_class == "Diffusion":
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "Diffusion":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda)


def make_fixed_lr_scheduler(optimizer):
    return LambdaLR(optimizer, lambda epoch: 1.0)


def make_scheduler(optimizer, num_steps):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_steps // 100, num_training_steps=num_steps
    )
    return scheduler


def get_image(ts, camera_names, crop_top=True, save_dir=None, t=None):
    curr_images = []
    for cam_name in camera_names:
        curr_image = ts.observation["images"][cam_name]

        # Check for 'cam_high' and apply transformation
        if crop_top and cam_name == "cam_high":
            curr_image = crop_resize(curr_image)

        # Swap BGR to RGB
        curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)

        curr_image = rearrange(curr_image, "h w c -> c h w")
        curr_images.append(curr_image)

    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().unsqueeze(0)

    # Center crop and resize
    original_size = curr_image.shape[-2:]
    ratio = 0.95
    curr_image = curr_image[
        ...,
        int(original_size[0] * (1 - ratio) / 2) : int(
            original_size[0] * (1 + ratio) / 2
        ),
        int(original_size[1] * (1 - ratio) / 2) : int(
            original_size[1] * (1 + ratio) / 2
        ),
    ]
    curr_image = curr_image.squeeze(0)
    resize_transform = transforms.Resize(original_size, antialias=True)
    curr_image = resize_transform(curr_image)
    curr_image = curr_image.unsqueeze(0)

    if save_dir is not None:
        # Convert torch tensors back to numpy and concatenate for visualization
        concat_images = [
            rearrange(img.numpy(), "c h w -> h w c")
            for img in curr_image.squeeze(0)
        ]
        concat_image = np.concatenate(concat_images, axis=1)
        concat_image = cv2.cvtColor(concat_image, cv2.COLOR_RGB2BGR)
        img_name = (
            "init_visualize.png" if t is None else f"gpt/{t=}.png"
        )  # save image every query_frequency for ChatGPT
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, (concat_image * 255).astype(np.uint8))

    return curr_image


def eval_ddpm(config, ckpt_name, save_episode=True, dataset_dirs=None):
    set_seed(1000)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    onscreen_cam = "cam_high"
    log_wandb = config["log_wandb"]

    # Load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    model_state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))["model_state_dict"]
    if is_multi_gpu_checkpoint(model_state_dict):
        print("The checkpoint was trained on multiple GPUs.")
        model_state_dict = {
            k.replace("module.", "", 1): v for k, v in model_state_dict.items()
        }
    loading_status = policy.deserialize(model_state_dict)
    print(loading_status)
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    if policy_class == "Diffusion":
        post_process = (
            lambda a: ((a + 1) / 2) * (stats["action_max"] - stats["action_min"])
            + stats["action_min"]
        )
    else:
        post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

    env = make_real_env(init_node=True)
    env_max_reward = 0

    query_frequency = policy_config["num_queries"]
    if temporal_agg:
        query_frequency = 25
        num_queries = policy_config["num_queries"]

    max_timesteps = int(max_timesteps * 1)  # May increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []

    n_existing_rollouts = (
        len([f for f in os.listdir(ckpt_dir) if f.startswith("video")])
        if save_episode
        else 0
    )
    print(f"{n_existing_rollouts=}")

    for rollout_id in range(num_rollouts):
        ts = env.reset()

        ### Onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(
                env._physics.render(height=480, width=640, camera_id=onscreen_cam)
            )
            plt.ion()

        ### Evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, state_dim]
            )

        qpos_history = torch.zeros((1, max_timesteps, state_dim))
        image_list = []  # For visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### Update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(
                        height=480, width=640, camera_id=onscreen_cam
                    )
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### Process previous timestep to get qpos and image_list
                obs = ts.observation
                if "images" in obs:
                    image_list.append(obs["images"])
                else:
                    image_list.append({"main": obs["image"]})
                qpos_numpy = np.array(obs["qpos"])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().unsqueeze(0)
                qpos_history[:, t] = qpos

                ### Query policy
                if policy_class in ["Diffusion"]:
                    if t % query_frequency == 0:
                        curr_image = get_image(
                            ts, camera_names, save_dir=ckpt_dir if t == 0 else None
                        )
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t : t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(
                            actions_for_curr_step != 0, axis=1
                        )
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (
                            torch.from_numpy(exp_weights).unsqueeze(dim=1)
                        )
                        raw_action = (actions_for_curr_step * exp_weights).sum(
                            dim=0, keepdim=True
                        )
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                
                else:
                    raise NotImplementedError

                ### Post-process actions
                raw_action = raw_action.squeeze(0).numpy()
                action = post_process(raw_action)
                # Only update if the absolute value of the action is greater than 0.1
                if np.any(np.abs(action) > 0.1):
                    target_qpos = action

                ts = env.step(target_qpos)

                ### For visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()

        move_grippers(
            [env.puppet_bot_left, env.puppet_bot_right],
            [PUPPET_GRIPPER_JOINT_OPEN] * 2,
            move_time=0.5,
        )  # Open grippers

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
        )
        if log_wandb:
            wandb.log(
                {
                    "test/episode_return": episode_return,
                    "test/episode_highest_reward": episode_highest_reward,
                    "test/env_max_reward": env_max_reward,
                    "test/success": episode_highest_reward == env_max_reward,
                },
                step=rollout_id,
            )

        if save_episode:
            video_name = f"video{rollout_id+n_existing_rollouts}.mp4"
            save_videos(
                image_list,
                DT,
                video_path=os.path.join(ckpt_dir, video_name),
                cam_names=camera_names,
            )
            if log_wandb:
                wandb.log(
                    {
                        "test/video": wandb.Video(
                            os.path.join(ckpt_dir, f"video{rollout_id}.mp4"),
                            fps=50,
                            format="mp4",
                        )
                    },
                    step=rollout_id,
                )

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

    print(summary_str)

    # Save success rate to txt
    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))

    if log_wandb:
        wandb.log({"test/success_rate": success_rate, "test/avg_return": avg_return})

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    return policy(qpos_data, image_data, action_data, is_pad)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', nargs='+', type=str, help='List of task names', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True) 
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    #diffusion
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--log_wandb', action='store_true')
    parser.add_argument('--gpu', action='store', type=int, help='gpu', default=0, required=False)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--max_skill_len', action='store', type=int, help='max_skill_len', required=False)

    args = parser.parse_args()
    config = vars(args)
    main(config)