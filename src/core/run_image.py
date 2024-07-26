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
import torch.nn.functional as F
from src.model.util import set_seed, detach_dict, compute_dict_mean
from environment.dataset.new_image_dataset import load_merged_data
from src.policy.image_ddpm import DiffusionPolicy
from src.model.util import is_multi_gpu_checkpoint, memory_monitor
from src.aloha.aloha_scripts.constants import DT, PUPPET_GRIPPER_JOINT_OPEN
from src.config.dataset_config import TASK_CONFIGS, DATA_DIR
# from src.aloha.aloha_scripts.real_env import make_real_env  
# from src.aloha.aloha_scripts.robot_utils import move_grippers
# from src.aloha.aloha_scripts.visualize_episodes import save_videos


CROP_TOP = False  # for aloha pro, whose top camera is high
CKPT = 0  # 0 for policy_last, otherwise put the ckpt number here

def signal_handler(sig,frame):
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
    history_len = args["history_len"]
    prediction_offset = args["prediction_offset"]
    history_skip_frame = args["history_skip_frame"]
    is_test = args["test"]

    # Set up wandb
    if log_wandb:
        if is_eval:
            # run_name += ".eval"
            log_wandb = False
        elif is_test:
            run_name = ckpt_dir.split("/")[-1] + f"seed.{args['seed']}.test"
            wandb.init(
                project="task1",
                entity="joeywang-of",
                name=run_name,
                config=args,
                resume="allow",
            )
        else:
            run_name = ckpt_dir.split("/")[-1] + f".{args['seed']}"
            wandb_run_id_path = os.path.join(ckpt_dir, "wandb_run_id.txt")
            # check if wandb run exists
            if os.path.exists(wandb_run_id_path):
                with open(wandb_run_id_path, "r") as f:
                    saved_run_id = f.read().strip()
                wandb.init(
                    project="task1",
                    entity="joeywang-of",
                    name=run_name,
                    resume=saved_run_id,
                )
            else:
                wandb.init(
                    project="task1",
                    entity="joeywang-of",
                    name=run_name,
                    config=args,
                    resume="allow",
                )
                # Ensure the directory exists before trying to open the file
                os.makedirs(os.path.dirname(wandb_run_id_path), exist_ok=True)
                with open(wandb_run_id_path, "w") as f:
                    f.write(wandb.run.id)

    if args["gpu"] is not None and not args["multi_gpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args['gpu']}"
        assert torch.cuda.is_available()

    # get task parameters
    dataset_dirs = []
    num_episodes_list = []
    max_episode_len = 0

    for task in task_name:
        
        task_config = TASK_CONFIGS[task]
        dataset_dirs.append(DATA_DIR)
        num_episodes_list.append(task_config["num_episodes"])
        max_episode_len = task_config["episode_len"]
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
        "observation_horizon": args["history_len"]+args["prediction_offset"]+1,
        "action_horizon": 16,  # TODO not used
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
        # results = []
        # for ckpt_name in ckpt_names:
        #     success_rate, avg_return = eval_ddpm(
        #         config, ckpt_name, save_episode=True, dataset_dirs=dataset_dirs
        #     )
        #     results.append([ckpt_name, success_rate, avg_return])

        # for ckpt_name, success_rate, avg_return in results:
        #     print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        # print()
        exit()


    
    train_dataloader, stats,val_dataloader,pretest_dataloader,test_dataloader= load_merged_data(
        dataset_dirs,
        num_episodes_list,
        camera_names,
        batch_size_train,
        max_len=max_skill_len,
        history_skip_frame=history_skip_frame,
        history_len=history_len,
        prediction_offset=prediction_offset,
        policy_class=policy_class,
    )
    if is_test:
        test_ddpm(test_dataloader, config, "policy_last.ckpt")
        exit()

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    train_ddpm(train_dataloader,val_dataloader,pretest_dataloader, config)


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
    # scheduler = make_fixed_lr_scheduler(optimizer)
    return scheduler


def get_image(ts, camera_names, crop_top=CROP_TOP, save_dir=None, t=None):
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
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

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
            rearrange(img.cpu().numpy(), "c h w -> h w c")
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


# def eval_ddpm(config:dict, ckpt_name, save_episode=True, dataset_dirs=None):
#     set_seed(1000)
#     ckpt_dir = config["ckpt_dir"]
#     state_dim = config["state_dim"]
#     policy_class = config["policy_class"]
#     onscreen_render = config["onscreen_render"]
#     policy_config = config["policy_config"]
#     camera_names = config["camera_names"]
#     max_timesteps = config["episode_len"]
#     temporal_agg = config["temporal_agg"]
#     onscreen_cam = "cam_high"
#     log_wandb = config["log_wandb"]

#     # Load policy and stats
#     ckpt_path = os.path.join(ckpt_dir, ckpt_name)
#     policy = make_policy(policy_class, policy_config)
#     model_state_dict = torch.load(ckpt_path)["model_state_dict"]
#     if is_multi_gpu_checkpoint(model_state_dict):
#         print("The checkpoint was trained on multiple GPUs.")
#         model_state_dict = {
#             k.replace("module.", "", 1): v for k, v in model_state_dict.items()
#         }
#     loading_status = policy.deserialize(model_state_dict)
#     print(loading_status)
#     policy.cuda()
#     policy.eval()
#     print(f"Loaded: {ckpt_path}")
#     stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
#     with open(stats_path, "rb") as f:
#         stats = pickle.load(f)

#     pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    
#     post_process = (
#             lambda a: ((a + 1) / 2) * (stats["action_max"] - stats["action_min"])
#             + stats["action_min"]
#         )
 
#     env = make_real_env(init_node=True)
#     env_max_reward = 0

#     query_frequency = policy_config["num_queries"]
#     if temporal_agg:
#         query_frequency = 25
#         num_queries = policy_config["num_queries"]

#     max_timesteps = int(max_timesteps * 1)  # May increase for real-world tasks

#     num_rollouts = 50
#     episode_returns = []
#     highest_rewards = []

#     n_existing_rollouts = (
#         len([f for f in os.listdir(ckpt_dir) if f.startswith("video")])
#         if save_episode
#         else 0
#     )
#     print(f"{n_existing_rollouts=}")

#     for rollout_id in range(num_rollouts):
#         ts = env.reset()

#         ### Onscreen render
#         if onscreen_render:
#             ax = plt.subplot()
#             plt_img = ax.imshow(
#                 env.physics.render(height=480, width=640, camera_id=onscreen_cam)
#             )
#             plt.ion()

#         ### Evaluation loop
#         if temporal_agg:
#             all_time_actions = torch.zeros(
#                 [max_timesteps, max_timesteps + num_queries, state_dim]
#             ).cuda()

#         qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
#         image_list = []  # For visualization
#         qpos_list = []
#         target_qpos_list = []
#         rewards = []
#         with torch.inference_mode():
#             for t in range(max_timesteps):
#                 ### Update onscreen render and wait for DT
#                 if onscreen_render:
#                     image = env.physics.render(
#                         height=480, width=640, camera_id=onscreen_cam
#                     )
#                     plt_img.set_data(image)
#                     plt.pause(DT)

#                 ### Process previous timestep to get qpos and image_list
#                 obs = ts.observation
#                 if "images" in obs:
#                     image_list.append(obs["images"])
#                 else:
#                     image_list.append({"main": obs["image"]})
#                 qpos_numpy = np.array(obs["qpos"])
#                 qpos = pre_process(qpos_numpy)
#                 qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
#                 qpos_history[:, t] = qpos

#                 ### Query policy
#                 if policy_class in ["Diffusion"]:
#                     if t % query_frequency == 0:
#                         curr_image = get_image(
#                             ts, camera_names, save_dir=ckpt_dir if t == 0 else None
#                         )
#                         all_actions = policy(qpos, curr_image)
#                     if temporal_agg:
#                         all_time_actions[[t], t : t + num_queries] = all_actions
#                         actions_for_curr_step = all_time_actions[:, t]
#                         actions_populated = torch.all(
#                             actions_for_curr_step != 0, axis=1
#                         )
#                         actions_for_curr_step = actions_for_curr_step[actions_populated]
#                         k = 0.01
#                         exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
#                         exp_weights = exp_weights / exp_weights.sum()
#                         exp_weights = (
#                             torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
#                         )
#                         raw_action = (actions_for_curr_step * exp_weights).sum(
#                             dim=0, keepdim=True
#                         )
#                     else:
#                         raw_action = all_actions[:, t % query_frequency]
                
#                 else:
#                     raise NotImplementedError

#                 ### Post-process actions
#                 raw_action = raw_action.squeeze(0).cpu().numpy()
#                 action = post_process(raw_action)
#                 # Only update if the absolute value of the action is greater than 0.1
#                 if np.any(np.abs(action) > 0.1):
#                     target_qpos = action

#                 ts = env.step(target_qpos)

#                 ### For visualization
#                 qpos_list.append(qpos_numpy)
#                 target_qpos_list.append(target_qpos)
#                 rewards.append(ts.reward)

#             plt.close()

#         move_grippers(
#             [env.puppet_bot_left, env.puppet_bot_right],
#             [PUPPET_GRIPPER_JOINT_OPEN] * 2,
#             move_time=0.5,
#         )  # Open grippers

#         rewards = np.array(rewards)
#         episode_return = np.sum(rewards[rewards != None])
#         episode_returns.append(episode_return)
#         episode_highest_reward = np.max(rewards)
#         highest_rewards.append(episode_highest_reward)
#         print(
#             f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}"
#         )
#         if log_wandb:
#             wandb.log(
#                 {
#                     "test/episode_return": episode_return,
#                     "test/episode_highest_reward": episode_highest_reward,
#                     "test/env_max_reward": env_max_reward,
#                     "test/success": episode_highest_reward == env_max_reward,
#                 },
#                 step=rollout_id,
#             )

#         if save_episode:
#             video_name = f"video{rollout_id+n_existing_rollouts}.mp4"
#             save_videos(
#                 image_list,
#                 DT,
#                 video_path=os.path.join(ckpt_dir, video_name),
#                 cam_names=camera_names,
#             )
#             if log_wandb:
#                 wandb.log(
#                     {
#                         "test/video": wandb.Video(
#                             os.path.join(ckpt_dir, f"video{rollout_id}.mp4"),
#                             fps=50,
#                             format="mp4",
#                         )
#                     },
#                     step=rollout_id,
#                 )

#     success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
#     avg_return = np.mean(episode_returns)
#     summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
#     for r in range(env_max_reward + 1):
#         more_or_equal_r = (np.array(highest_rewards) >= r).sum()
#         more_or_equal_r_rate = more_or_equal_r / num_rollouts
#         summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

#     print(summary_str)

#     # Save success rate to txt
#     result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
#     with open(os.path.join(ckpt_dir, result_file_name), "w") as f:
#         f.write(summary_str)
#         f.write(repr(episode_returns))
#         f.write("\n\n")
#         f.write(repr(highest_rewards))

#     if log_wandb:
#         wandb.log({"test/success_rate": success_rate, "test/avg_return": avg_return})

#     return success_rate, avg_return


def forward_pass(data, policy):
    image_data, action_data, is_pad = data
    image_data, action_data, is_pad = (
        image_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )
    return policy( image_data, action_data, is_pad)


def train_ddpm(train_dataloader, val_dataloader, pretest_dataloader, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    log_wandb = config["log_wandb"]
    multi_gpu = config["policy_config"]["multi_gpu"]

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    optimizer = make_optimizer(policy_class, policy)
    scheduler = make_scheduler(optimizer, num_epochs)

    # if ckpt_dir is not empty, prompt the user to load the checkpoint
    if os.path.isdir(ckpt_dir) and len(os.listdir(ckpt_dir)) > 2:
        print(f"Checkpoint directory {ckpt_dir} is not empty. Load checkpoint? (y/n)")
        load_ckpt = input()
        if load_ckpt == "y":
            # load the latest checkpoint
            latest_idx = max(
                [
                    int(f.split("_")[2])
                    for f in os.listdir(ckpt_dir)
                    if f.startswith("policy_epoch_")
                ]
            )
            ckpt_path = os.path.join(
                ckpt_dir, f"policy_epoch_{latest_idx}_seed_{seed}.ckpt"
            )
            print(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path)
            model_state_dict = checkpoint["model_state_dict"]
            # The model was trained on a single gpu, now load onto multiple gpus
            if multi_gpu and not is_multi_gpu_checkpoint(model_state_dict):
                # Add "module." prefix only to the keys associated with policy.model
                model_state_dict = {
                    k if "model" not in k else f"model.module.{k.split('.', 1)[1]}": v
                    for k, v in model_state_dict.items()
                }
            # The model was trained on multiple gpus, now load onto a single gpu
            elif not multi_gpu and is_multi_gpu_checkpoint(model_state_dict):
                # Remove "module." prefix only to the keys associated with policy.model
                model_state_dict = {
                    k.replace("module.", "", 1): v for k, v in model_state_dict.items()
                }
            loading_status = policy.deserialize(model_state_dict)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(loading_status)
        else:
            print("Not loading checkpoint")
            start_epoch = 0
    else:
        start_epoch = 0

    policy.cuda()
    best_val_loss = float('inf')
    train_history = []
    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f"\nEpoch {epoch}")
        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            try:
                forward_dict = forward_pass(data, policy)
                # backward
                loss = forward_dict["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_history.append(detach_dict(forward_dict))
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    for p in policy.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                else:
                    raise 
        scheduler.step()
        e = epoch - start_epoch
        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * e : (batch_idx + 1) * (e + 1)]
        )
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        epoch_summary["lr"] = np.array(scheduler.get_last_lr()[0])
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.5f} "
        print(summary_string)
        if log_wandb:
            epoch_summary_train = {f"train/{k}": v for k, v in epoch_summary.items()}
            wandb.log(epoch_summary_train, step=epoch)
        
        # validation
        policy.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                try:
                    # image_data, qpos_data, action_data, is_pad = [item.cuda() for item in batch]
                    # outputs = policy(qpos_data, image_data)
                    forward_dict = forward_pass(batch, policy)
                    loss = forward_dict["loss"]
                    val_loss += loss.item()
                except:
                    if 'out of memory' in str(e):
                        print('| WARNING: ran out of memory during validation, skipping batch')
                        torch.cuda.empty_cache()
                    else:
                        raise 

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(policy.state_dict(), 'best_model.pth')
            print("Model saved!")

        if log_wandb:
            wandb.log({"val/loss": avg_val_loss}, step=epoch)

        save_ckpt_every = 100
        if epoch % save_ckpt_every == 0 and epoch > 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(
                {
                    "model_state_dict": policy.serialize(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path,
            )

            # Pruning: this removes the checkpoint save_ckpt_every epochs behind the current one
            # except for the ones at multiples of 1000 epochs
            prune_epoch = epoch - save_ckpt_every
            if prune_epoch % 1000 != 0:
                prune_path = os.path.join(
                    ckpt_dir, f"policy_epoch_{prune_epoch}_seed_{seed}.ckpt"
                )
                if os.path.exists(prune_path):
                    os.remove(prune_path)

        save_ckpt_every = 100
        if epoch % save_ckpt_every == 0 and epoch > 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(
                {
                    "model_state_dict": policy.serialize(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path,
            )

            # Pruning: this removes the checkpoint save_ckpt_every epochs behind the current one
            # except for the ones at multiples of 1000 epochs
            prune_epoch = epoch - save_ckpt_every
            if prune_epoch % 1000 != 0:
                prune_path = os.path.join(
                    ckpt_dir, f"policy_epoch_{prune_epoch}_seed_{seed}.ckpt"
                )
                if os.path.exists(prune_path):
                    os.remove(prune_path)

    # test
    policy.load_state_dict(torch.load('best_model.pth'))
    policy.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in tqdm(pretest_dataloader):
            try:
                # image_data, qpos_data, action_data, is_pad = [item.cuda() for item in batch]
                # outputs = policy(image_data, qpos_data)
                forward_dict = forward_pass(batch, policy)
                loss = forward_dict["loss"]
                test_loss += loss.item()
            except:
                if 'out of memory' in str(e):
                        print('| WARNING: ran out of memory during validation, skipping batch')
                        torch.cuda.empty_cache()
                else:
                    raise 

    avg_test_loss = test_loss / len(pretest_dataloader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    if log_wandb:
        wandb.log({"test/loss": avg_test_loss})

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(
        {
            "model_state_dict": policy.serialize(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        },
        ckpt_path,
    )
    

def test_ddpm(test_dataloader, config, ckpt_name):
    ckpt_dir = config["ckpt_dir"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]

    # Load policy
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    model_state_dict = torch.load(ckpt_path)["model_state_dict"]
    loading_status = policy.deserialize(model_state_dict)
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {ckpt_path}")
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    post_process = (
            lambda a: ((a + 1) / 2) * (stats["action_max"] - stats["action_min"])
            + stats["action_min"]
        )
    
    # START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239, 0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
    # Create save directory
    save_dir = os.path.join(ckpt_dir, "test_results")
    os.makedirs(save_dir, exist_ok=True)

    def plot_joint_positions(pred_actions, true_actions, idx):
        timesteps = np.arange(true_actions.shape[0])
        fig, axes = plt.subplots(5, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(14):
            ax = axes[i]
            ax.plot(timesteps, pred_actions[:true_actions.shape[0], i], label='Predicted', color='red', linestyle='--')
            ax.plot(timesteps, true_actions[:, i], label='True', color='blue')
            ax.set_title(f'Joint {i+1}')
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Position')
            ax.grid(True)
            if i == 0:  # Add legend to the first subplot
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'joint_positions_{idx}.png'))
        plt.close()

    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_dataloader)):
            images, true_actions, _ = [item.cuda() for item in data]
            predicted_actions = policy(images)
            
            # Assuming the predicted_actions shape is [batch_size, T, action_dim]
            batch_size = predicted_actions.shape[0]
            num_batches += batch_size

            for i in range(batch_size):
                pred_actions = post_process(predicted_actions[i].cpu().numpy())
                true_actions_np = post_process(true_actions[i].cpu().numpy())
                
                plot_joint_positions(pred_actions, true_actions_np, idx * batch_size + i)
                
                # Compute loss
                loss = F.mse_loss(torch.tensor(pred_actions[:true_actions_np.shape[0]]), torch.tensor(true_actions_np))
                total_loss += loss.item()
                print(f"batch_loss:{loss}")

    avg_loss = total_loss / num_batches
    print(f"Testing completed. Results saved in: {save_dir}")
    print(f"Average MSE Loss: {avg_loss:.4f}")

    

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
    parser.add_argument('--history_len', type=int, default=2)
    parser.add_argument('--prediction_offset', type=int, default=5)
    parser.add_argument('--history_skip_frame', type=int, default=10)
    parser.add_argument('--test', action='store_true',default=False)
    args = parser.parse_args()
    config = vars(args)
    main(config)