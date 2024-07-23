import argparse
import math
import wandb
import torch
import numpy as np
import threading
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from torch.optim.lr_scheduler import LambdaLR
import signal
import cv2
import os
import torch.nn.functional as F
from src.model.util import set_seed, detach_dict, compute_dict_mean
from environment.dataset.test_image_dataset import load_merged_data
from src.policy.image_ddpm import DiffusionPolicy
from src.model.util import is_multi_gpu_checkpoint, memory_monitor
from src.config.dataset_config import TASK_CONFIGS, DATA_DIR
from src.aloha.aloha_scripts.visualize_episodes import STATE_NAMES


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
        if is_test:
            log_wandb = False
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

def forward_pass(data, policy):
    image_data, action_data, is_pad = data
    image_data, action_data, is_pad = (
        image_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )
    return policy( image_data, action_data, is_pad)

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

    # Load stats for post-processing actions
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    post_process = (
        lambda a: ((a + 1) / 2) * (stats["action_max"] - stats["action_min"])
        + stats["action_min"]
    )

    # Create save directory
    save_dir = os.path.join(ckpt_dir, "test_results")
    os.makedirs(save_dir, exist_ok=True)

    def plot_joint_positions(pred_actions, true_actions, idx):
        timesteps = np.arange(true_actions.shape[0])
        fig, axes = plt.subplots(5, 3, figsize=(15, 10))
        axes = axes.flatten()
        all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
        for i in range(14):
            ax = axes[i]
            ax.plot(timesteps, true_actions[:, i], label='True', color='blue')
            ax.plot(timesteps, pred_actions[:true_actions.shape[0], i], label='Predicted', color='red', linestyle='--')
            ax.set_title(f'Joint {i}: {all_names[i]}')
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Position')
            ax.grid(True)
            if i == 0:  # Add legend to the first subplot
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'joint_positions_{idx}.png'))
        plt.close()

    total_loss = 0
    num_episodes = 0

    with torch.no_grad():
        for episode_idx, episode_data in enumerate(tqdm(test_dataloader)):
            images, true_actions, is_pad = [item.cuda() for item in episode_data]
            
            # Process the entire episode
            episode_length = (~is_pad).sum().item()
            images = images[:episode_length]
            true_actions = true_actions[:episode_length]
            
            # Predict actions for the entire episode
            predicted_actions = []
            window_size = policy_config["observation_horizon"]
            stride = policy_config["prediction_horizon"]
            
            for i in range(0, episode_length, stride):
                window = images[i:i+window_size]
                if window.shape[0] < window_size:
                    # Pad the last window if necessary
                    padding = torch.zeros((window_size - window.shape[0],) + window.shape[1:], device=window.device)
                    window = torch.cat([window, padding], dim=0)
                
                window_pred = policy(window.unsqueeze(0))
                predicted_actions.append(window_pred[0, :min(stride, episode_length-i)])
            
            predicted_actions = torch.cat(predicted_actions, dim=0)
            
            # Post-process actions
            pred_actions = post_process(predicted_actions.cpu().numpy())
            true_actions_np = post_process(true_actions.cpu().numpy())
            
            # Plot results
            plot_joint_positions(pred_actions, true_actions_np, episode_idx)
            
            # Compute loss
            loss = F.mse_loss(torch.tensor(pred_actions), torch.tensor(true_actions_np))
            total_loss += loss.item()
            num_episodes += 1
            
            print(f"Episode {episode_idx}, MSE Loss: {loss:.4f}")

    avg_loss = total_loss / num_episodes
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