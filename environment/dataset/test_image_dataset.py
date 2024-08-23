import numpy as np
import torch
import os
import h5py
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.utils.data
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from src.model.util import DAggerSampler

CROP_TOP = False  # hardcode

import numpy as np
import torch
import os
import h5py
import cv2
from torch.utils.data import Dataset

class TestDataset(Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        max_len=None,
        policy_class=None,
    ):
        super().__init__()
        self.episode_ids = episode_ids if len(episode_ids) > 0 else [0]
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.max_len = max_len
        self.policy_class = policy_class
        self.transformations = None
        self.episode_lengths = self._get_episode_lengths()
        self.cumulative_lengths = np.cumsum([0] + self.episode_lengths)

    def _get_episode_lengths(self):
        lengths = []
        for episode_id in self.episode_ids:
            dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
            with h5py.File(dataset_path, "r") as root:
                lengths.append(root["/action"].shape[0])
        return lengths

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, index):
        episode_index = np.searchsorted(self.cumulative_lengths, index, side='right') - 1
        episode_id = self.episode_ids[episode_index]
        timestep = index - self.cumulative_lengths[episode_index]
        
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        
        with h5py.File(dataset_path, "r") as root:
            compressed = root.attrs.get("compress", False)
            end_ts = min(timestep + self.max_len, root["/action"].shape[0]) if self.max_len else root["/action"].shape[0]

            # Load and process images
            image_dict = {cam: root[f"/observations/images/{cam}"][timestep] for cam in self.camera_names}
            if compressed:
                for cam in image_dict:
                    image = cv2.imdecode(image_dict[cam], 1)
                    image_dict[cam] = np.array(image)

            # Convert images to RGB and stack them
            all_cam_images = np.stack([cv2.cvtColor(image_dict[cam], cv2.COLOR_BGR2RGB) for cam in self.camera_names])
            image_data = torch.einsum("k h w c -> k c h w", torch.from_numpy(all_cam_images / 255.0))

            # Process actions and padding
            action = root["/action"][timestep:end_ts]
            padded_action = np.zeros((self.max_len,) + action.shape[1:], dtype=np.float32)
            padded_action[:len(action)] = action
            is_pad = torch.from_numpy(np.pad(np.ones(len(action)), (0, self.max_len - len(action)), 'constant')).bool()

            if self.policy_class == "Diffusion":
                action_data = 2 * (torch.from_numpy(padded_action) - self.norm_stats["action_min"]) / \
                              (self.norm_stats["action_max"] - self.norm_stats["action_min"]) - 1
            else:
                action_data = (torch.from_numpy(padded_action) - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

            # Initialize transformations if needed
            if self.transformations is None:
                self.initialize_transformations(image_data.shape[2:])

            for transform in self.transformations:
                image_data = transform(image_data).float()

            return image_data, action_data, is_pad
    def initialize_transformations(self, original_size):
        ratio = 0.95
        self.transformations = [
            transforms.RandomCrop([int(original_size[0] * ratio), int(original_size[1] * ratio)]),
            transforms.Resize(original_size, antialias=True)
        ]
        if self.policy_class == "Diffusion":
            self.transformations.extend([
                transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
            ])
        
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
    dagger_ratio=0.9,
    policy_class=None,
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

    norm_stats = TestDataset.get_norm_stats(dataset_dirs, num_episodes_list)

    train_ratio = 0.8
    val_ratio = 0.1

    shuffled_indices = np.random.permutation(all_episode_indices)
    train_split = int(train_ratio * len(all_episode_indices))
    val_split = int((train_ratio + val_ratio) * len(all_episode_indices))
    
    train_indices = shuffled_indices[:train_split]
    val_indices = shuffled_indices[train_split:val_split]
    pretest_indices = shuffled_indices[val_split:]

    train_datasets = [
        TestDataset(
            [idx for d, idx in train_indices if d == dataset_dir],
            dataset_dir,
            camera_names,
            norm_stats,
            max_len,
            policy_class=policy_class,
        )
        for dataset_dir in dataset_dirs
    ]
    val_datasets = [
        TestDataset(
            [idx for d, idx in val_indices if d == dataset_dir], 
            dataset_dir, 
            camera_names, 
            norm_stats,
            max_len, 
            policy_class=policy_class
        ) 
        for dataset_dir in dataset_dirs
    ]
    
    pretest_datasets = [
        TestDataset(
            [idx for d, idx in pretest_indices if d == dataset_dir],
            dataset_dir,
            camera_names,
            norm_stats,
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

    if dagger_ratio is not None:
        dataset_sizes = {
            dataset_dir: num_episodes
            for dataset_dir, num_episodes in zip(dataset_dirs, num_episodes_list)
        }
        dagger_sampler = DAggerSampler(
            all_episode_indices,
            last_dataset_indices,
            batch_size_train,
            dagger_ratio,
            dataset_sizes,
        )
        train_dataloader = DataLoader(
            merged_train_dataset,
            batch_sampler=dagger_sampler,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=4,
            persistent_workers=True,
        )
        val_dataloader = DataLoader(
            merged_val_dataset,
            batch_size=batch_size_train,
            pin_memory=True,
            num_workers=2,
            prefetch_factor=4,
            persistent_workers=True,
        )
    else:
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
        num_workers=2,
        prefetch_factor=1,
    )
    test_dataloader = DataLoader(
        merged_test_dataset,
        batch_size=1,  # Set batch size to 1 to return one episode at a time
        shuffle=False,
        pin_memory=True,
        num_workers=20,
        prefetch_factor=8,
    )
    return train_dataloader, norm_stats, val_dataloader, pretest_dataloader, test_dataloader
