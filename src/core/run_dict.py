import argparse
import math
import wandb
import torch
import numpy as np
import threading
from tqdm import tqdm
import pickle
from torch.optim.lr_scheduler import LambdaLR
import signal
import time
import os
from src.model.util import set_seed, detach_dict, compute_dict_mean
from environment.dataset.dict_image_dataset import load_merged_data
from src.policy.ddpm_dict import DiffusionPolicy
from src.model.util import is_multi_gpu_checkpoint, memory_monitor
from src.config.dataset_config import TASK_CONFIGS, DATA_DIR

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
    is_test = args["test"]

    # Set up wandb
    if log_wandb:
        if is_eval:
            # run_name += ".eval"
            log_wandb = False
        elif is_test:
            run_name = ckpt_dir.split("/")[-1] + f"localseed.{args['seed']}.{time.time()}.test"
            wandb.init(
                project="task1",
                entity="joeywang-of",
                name=run_name,
                config=args,
                resume="allow",
                mode="disabled",
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
        max_episode_len = max(max_episode_len, task_config["episode_len"])
        camera_names = task_config["camera_names"]


    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5

    policy_config = {
        "lr": args["lr"],
        "camera_names": camera_names,
        "action_dim": 14,
        "observation_horizon": args["history_len"]+1,
        "action_horizon": 8,  # TODO 
        "prediction_horizon": args["history_len"]+args["prediction_offset"]+1,
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
        }

    train_dataloader, stats,val_dataloader,pretest_dataloader,test_dataloader= load_merged_data(
        dataset_dirs,
        num_episodes_list,
        camera_names,
        batch_size_train,
        max_len=history_len+prediction_offset+1,
        history_len=history_len,
        prediction_offset=prediction_offset,
        policy_class=policy_class,
    )


    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    train_ddpm(test_dataloader,val_dataloader,pretest_dataloader, config)


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
        {k: v.cuda() for k, v in image_data.items()},
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
            best_model_path = os.path.join(ckpt_dir, f"best_model_epoch_{epoch}_seed_{seed}.pth")
            torch.save(policy.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path}")


        if log_wandb:
            wandb.log({"val/loss": avg_val_loss}, step=epoch)

        save_ckpt_every = 10
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
            if prune_epoch % 50 != 0:
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
    parser.add_argument('--history_len', type=int, default=1)
    parser.add_argument('--prediction_offset', type=int, default=14)
    parser.add_argument('--test', action='store_true',default=False)
    args = parser.parse_args()
    config = vars(args)
    main(config)