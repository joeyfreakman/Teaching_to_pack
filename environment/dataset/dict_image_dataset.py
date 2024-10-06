import numpy as np
import torch
import os
import h5py
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from src.model.util import DAggerSampler
import argparse
from torchvision import transforms
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.aloha.aloha_scripts.visualize_episodes import visualize_joints,load_hdf5

class Dictimagedataset(Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        history_len: int,
        prediction_offset: int,
        max_len=None,
        policy_class=None,
        ):
        super().__init__()
        self.episode_ids = episode_ids if len(episode_ids) > 0 else [0]
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.history_len = history_len
        self.prediction_offset = prediction_offset
        self.max_len = history_len + prediction_offset + 1
        self.policy_class = policy_class
        self.transformations = None
        self.obs_horizon = history_len + 1
        self.episode_starts = [0]
        self.total_len = 0
        for episode_id in self.episode_ids:
            with h5py.File(os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5"), "r") as f:
                episode_len = len(f["/action"])
                self.total_len += episode_len
                self.episode_starts.append(self.total_len)
        if self.transformations is None:
            original_size = (288,384)
            ratio = 0.95
            self.transformations= transforms.Compose([
    
                transforms.Resize(original_size,antialias=True),
            ])
            if self.policy_class == "Diffusion":
                self.transformations.transforms.extend(
                    [
                        transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                        # transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
                    ]
                )

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        episode_idx = 0
        while index >= self.episode_starts[episode_idx + 1]:
            episode_idx += 1
        local_index = index - self.episode_starts[episode_idx]

        episode_id = self.episode_ids[episode_idx]
        with h5py.File(os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5"), "r") as f:
            compressed = f.attrs.get("compress", False)
            # Get episode length
            episode_length = len(f["/action"])
            # Calculate max start
            max_start = min(local_index, episode_length - self.obs_horizon)

            image_dict = {}
            for cam_name in self.camera_names:
                cam_images = f[f"/observations/images/{cam_name}"][max_start:max_start + self.obs_horizon]
                
                if compressed:
                    cam_images = np.stack([cv2.cvtColor(cv2.imdecode(img, 1), cv2.COLOR_BGR2RGB) for img in cam_images])
                else:
                    cam_images = np.array(cam_images)

                image_dict[cam_name] = torch.from_numpy(cam_images).float().permute(0, 3, 1, 2)
                image_dict[cam_name]= self.transformations(image_dict[cam_name])

                image_dict[cam_name] = image_dict[cam_name] / 255.0
            
            assert all(image_dict[cam].shape == image_dict[self.camera_names[0]].shape for cam in self.camera_names), "All cameras should have the same image dimensions"

            action_sequence = f["/action"][max_start:max_start + self.max_len]

        action_data = torch.from_numpy(action_sequence).float()
        action_len = len(action_sequence)
        if action_len < self.max_len:
            padded_action = torch.zeros((self.max_len, action_data.shape[1]), dtype=torch.float32)
            padded_action[:action_len] = action_data
            padded_action[action_len:] = action_data[-1]
            action_data = padded_action

        is_pad = torch.zeros(self.max_len, dtype=torch.bool)
        is_pad[action_len:] = True
                
        action_data = (action_data - self.norm_stats["action_min"]) / (self.norm_stats["action_max"] - self.norm_stats["action_min"]) * 2 - 1
        
        return image_dict, action_data, is_pad
    

    @staticmethod
    def get_norm_stats(dataset_dirs, num_episodes_list):
        all_action_data = []

        for dataset_dir, num_episodes in zip(dataset_dirs, num_episodes_list):
            for episode_idx in range(num_episodes):
                dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
                with h5py.File(dataset_path, "r") as root:
                    action = root["/action"][()]
                all_action_data.append(torch.from_numpy(action))

        all_action_data = torch.cat(all_action_data, dim=0)

        action_mean = all_action_data.mean(dim=[0]).float()
        action_std = all_action_data.std(dim=[0]).float()
        action_std = torch.clip(action_std, 1e-2, np.inf)

        action_min = all_action_data.min(dim=0).values.float()
        action_max = all_action_data.max(dim=0).values.float()
        eps = 0.0001

        stats = {
            "action_mean": action_mean.numpy(),
            "action_std": action_std.numpy(),
            "action_min": action_min.numpy() - eps,
            "action_max": action_max.numpy() + eps,
        }

        return stats
    

    ### Merge multiple datasets
def load_merged_data(
    dataset_dirs,
    num_episodes_list,
    camera_names,
    batch_size_train,
    max_len=None,
    dagger_ratio=None,
    policy_class=None,
    history_len=1,
    prediction_offset=14,
):
    assert len(dataset_dirs) == len(
        num_episodes_list
    ), "Length of dataset_dirs and num_episodes_list must be the same."
    if dagger_ratio is not None:
        assert 0 <= dagger_ratio <= 1, "dagger_ratio must be between 0 and 1."

    all_episode_indices = []
    last_dataset_indices = []

    for i, (dataset_dir, num_episodes) in enumerate(
        zip(dataset_dirs, num_episodes_list)
    ):
        print(f"\nData from: {dataset_dir}\n")

        episode_indices = [(dataset_dir, i) for i in range(num_episodes)]

        if i == len(dataset_dirs) - 1:  # Last dataset
            last_dataset_indices.extend(episode_indices)
        all_episode_indices.extend(episode_indices)

    norm_stats = Dictimagedataset.get_norm_stats(dataset_dirs, num_episodes_list)

    train_ratio = 0.8
    val_ratio = 0.1

    shuffled_indices = np.random.permutation(all_episode_indices)
    train_split = int(train_ratio * len(all_episode_indices))
    val_split = int((train_ratio + val_ratio) * len(all_episode_indices))
    
    train_indices = shuffled_indices[:train_split]
    val_indices = shuffled_indices[train_split:val_split]
    pretest_indices = shuffled_indices[val_split:]

    train_datasets = [
        Dictimagedataset(
            [idx for d, idx in train_indices if d == dataset_dir],
            dataset_dir,
            camera_names,
            norm_stats,
            history_len,
            prediction_offset,
            max_len,
            policy_class=policy_class,
        )
        for dataset_dir in dataset_dirs
    ]
    val_datasets = [
        Dictimagedataset(
            [idx for d, idx in val_indices if d == dataset_dir], 
            dataset_dir, 
            camera_names, 
            norm_stats,
            history_len,
            prediction_offset,
            max_len, 
            policy_class=policy_class
        ) 
        for dataset_dir in dataset_dirs
    ]
    
    pretest_datasets = [
        Dictimagedataset(
            [idx for d, idx in pretest_indices if d == dataset_dir],
            dataset_dir,
            camera_names,
            norm_stats,
            history_len,
            prediction_offset,
            max_len,
            policy_class=policy_class,
        )
        for dataset_dir in dataset_dirs
    ]
    for dataset in train_datasets + val_datasets + pretest_datasets:
        if len(dataset) == 0:
            print(f"Warning: Empty dataset found in {dataset.dataset_dir}")

    test_datasets = train_datasets + val_datasets + pretest_datasets
    merged_train_dataset = ConcatDataset(train_datasets)
    merged_val_dataset = ConcatDataset(val_datasets)
    merged_pretest_dataset = ConcatDataset(pretest_datasets)
    merged_test_dataset = ConcatDataset(test_datasets)

    train_dataloader = DataLoader(
        merged_train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        merged_val_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,
    )
    pretest_dataloader = DataLoader(
        merged_pretest_dataset,
        batch_size=batch_size_train,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=1,
    )
    test_dataloader = DataLoader(
        merged_test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=1,
    )
    return train_dataloader, norm_stats, val_dataloader, pretest_dataloader, test_dataloader

"""
Test the Dataset class.

Example usage:
$ python dict_image_dataset.py --dataset_dir /mnt/d/kit/ALR/dataset/ttp_compressed/
"""
if __name__ == "__main__":
    t_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to the dataset directory"
    )
    args = parser.parse_args()

    camera_names = ["cam_high","cam_left_wrist", "cam_low",  "cam_right_wrist"]
    history_len = 1
    prediction_offset = 14
    num_episodes = 50  # Just to sample from the first 50 episodes for testing
    norm_stats = Dictimagedataset.get_norm_stats([args.dataset_dir], [num_episodes])
    max_len = history_len + prediction_offset + 1
    obs_len = history_len + 1
    dataset = Dictimagedataset(
        list(range(num_episodes)),
        args.dataset_dir,
        camera_names,
        norm_stats,
        history_len,
        prediction_offset,
        max_len,
        policy_class="Diffusion",
    )
    stats = Dictimagedataset.get_norm_stats([args.dataset_dir], [num_episodes])
    dataset_name = 'episode_0'
    save_dir = os.path.join(args.dataset_dir, "test")
    post_process = (
        lambda a: ((a + 1) / 2) * (stats["action_max"] - stats["action_min"])
        + stats["action_min"]
    )
    # idx=0
    # image_data, action_data, is_pad = dataset[idx]
    # true_action = post_process(action_data.numpy())

    # _, _, _, original_action, _ = load_hdf5(args.dataset_dir, dataset_name)

    # # 确保true_action长度与原始action相同
    # original_action = original_action[idx:len(true_action)+idx]

    # visualize_joints(original_action, true_action, plot_path=save_dir)
    # print(f"Testing completed. Results saved in: {save_dir}")

    # tru_traj = []
    # for i in range(0, 750,16):  # 每16步取一次数据
    #     image_data, action_data, is_pad = dataset[i]  # 使用整除来获取正确的索引
    #     tru_traj.extend(action_data)
    # tru_traj = post_process(np.array(tru_traj))
    # # 确保轨迹长度为750
    # tru_traj = np.array(tru_traj[:750])


    # # 加载原始数据进行比较
    # _, _, original_action, _ = load_hdf5(args.dataset_dir, dataset_name)

    # # 确保original_action长度与tru_traj相同
    # original_action = original_action[:750]

    
    # visualize_joints(original_action, tru_traj, plot_path=save_dir)
    # print(f"Testing completed. Results saved in: {save_dir}")

    # tru_traj = []
    # _, _, _, original_action, _ = load_hdf5(args.dataset_dir, dataset_name)
    # test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # for idx, data in enumerate(test_dataloader):

    #     image_data, action_data, is_pad = data
    #     true_action = post_process(action_data[0])

    #     while idx <750:
    #         if  idx ==1 or idx % 16 == 0:

    #             tru_traj.extend(true_action)

    #         if len(tru_traj)>=750:
    #             true_traj = np.array(tru_traj[:750])
    #             visualize_joints(original_action, true_traj, plot_path=save_dir)
    #             print(f"Testing completed. Results saved in: {save_dir}")
    #             break

        
    idx = np.random.randint(0, len(dataset))
    print(f"dataset_len: {len(dataset)}")
    image_sequence, action_data, is_pad = dataset[idx]
    print(f"Sampled dataset index: {idx}")
    
    num = image_sequence['cam_high'].shape[0]
    output_dir = os.path.join(dataset.dataset_dir,"plot")
    os.makedirs(output_dir, exist_ok=True)
    
    for t in tqdm(range(num)):
        plt.figure(figsize=(10, 5))
        for cam_idx, cam_name in enumerate(camera_names):
            plt.subplot(1, len(camera_names), cam_idx + 1)
            img_rgb = cv2.cvtColor(
                image_sequence[cam_name][t].permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB
            )
            plt.imshow(img_rgb)
            plt.title(f"{cam_name} at timestep {t}")
    
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"image_sequence_timestep_{t}.png"))
        print(f"Saved image_sequence_timestep_{t}.png")
        plt.close()
    print(f"Time taken: {time.time() - t_start:.5f}s")