import numpy as np
import torch
import os
import h5py
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.utils.data
import cv2
from scripts.data_pruning import crop_resize
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from src.model.util import DAggerSampler

CROP_TOP = False  # hardcode

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        history_skip_frame: int,
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
        self.max_len = history_len + prediction_offset + 1
        self.policy_class = policy_class
        self.transformations = None
        self.history_len = history_len
        self.prediction_offset = prediction_offset
        self.history_skip_frame = history_skip_frame

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)):
            raise TypeError(f"Index must be an integer, got {type(index)}")
        index = int(index)
        if index < 0 or index >= len(self.episode_ids):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.episode_ids)}")

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")

        with h5py.File(dataset_path, "r") as root:
            compressed = root.attrs.get("compress", False)
            total_timesteps = root["/action"].shape[0]
            
            curr_ts = np.random.randint(
                    self.history_len * self.history_skip_frame,
                    total_timesteps - self.prediction_offset,
                )
            start_ts = curr_ts - self.history_len * self.history_skip_frame
            target_ts = curr_ts + self.prediction_offset
            
            image_sequence = []
            action_sequence = []

            # Get images and actions for history
            for t in range(start_ts, curr_ts, self.history_skip_frame):
                if (t >= 0) and (t < total_timesteps):
                    image_dict = {cam_name: root[f"/observations/images/{cam_name}"][t] for cam_name in self.camera_names}
                    if compressed:
                        for cam_name in image_dict.keys():
                            decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                            image_dict[cam_name] = np.array(decompressed_image)
                            if CROP_TOP and cam_name == "cam_high":
                                image_dict[cam_name] = crop_resize(image_dict[cam_name])
                    for cam_name in image_dict.keys():
                        image_dict[cam_name] = cv2.cvtColor(image_dict[cam_name], cv2.COLOR_BGR2RGB)
                    all_cam_images = [image_dict[cam_name] for cam_name in self.camera_names]
                    all_cam_images = np.stack(all_cam_images, axis=0)
                    image_sequence.append(all_cam_images)
                    action_sequence.append(root["/action"][t])
            


            # Get images and actions for prediction offset
            for t in range(curr_ts, target_ts + 1):
                if (t >= 0) and (t < total_timesteps):
                    image_dict = {cam_name: root[f"/observations/images/{cam_name}"][t] for cam_name in self.camera_names}
                    if compressed:
                        for cam_name in image_dict.keys():
                            decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                            image_dict[cam_name] = np.array(decompressed_image)
                            if CROP_TOP and cam_name == "cam_high":
                                image_dict[cam_name] = crop_resize(image_dict[cam_name])
                    for cam_name in image_dict.keys():
                        image_dict[cam_name] = cv2.cvtColor(image_dict[cam_name], cv2.COLOR_BGR2RGB)
                    all_cam_images = [image_dict[cam_name] for cam_name in self.camera_names]
                    all_cam_images = np.stack(all_cam_images, axis=0)
                    image_sequence.append(all_cam_images)
                    action_sequence.append(root["/action"][t])
            
            image_sequence = np.stack(image_sequence, axis=0)
            image_data = torch.from_numpy(image_sequence)
            image_data = torch.einsum("t k h w c -> t k c h w", image_data)

            action_sequence = np.stack(action_sequence, axis=0)
            action_data = torch.from_numpy(action_sequence).float()

            action_len = len(action_sequence)
            if action_len > self.max_len:
                action_sequence = action_sequence[:self.max_len]
                action_len = self.max_len

            padded_action = np.zeros((self.max_len,) + action_sequence.shape[1:], dtype=np.float32)
            padded_action[:action_len] = action_sequence
            action_data = torch.from_numpy(padded_action).float()

            is_pad = torch.zeros(self.max_len, dtype=torch.bool)
            is_pad[action_len:] = True

            if self.transformations is None:
                original_size = image_data.shape[3:]
                ratio = 0.95
                self.transformations = [
                    transforms.RandomCrop(
                        size=[
                            int(original_size[0] * ratio),
                            int(original_size[1] * ratio),
                        ]
                    ),
                    transforms.Resize(original_size, antialias=True),
                ]
                if self.policy_class == "Diffusion":
                    self.transformations.extend(
                        [
                            transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                            transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
                        ]
                    )

            for transform in self.transformations:
                image_data = torch.stack([transform(img) for img in image_data])

            image_data = image_data / 255.0

            action_data = (
                (action_data - self.norm_stats["action_min"])
                / (self.norm_stats["action_max"] - self.norm_stats["action_min"])
            ) * 2 - 1
            
            return image_data, action_data, is_pad
        
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
    history_skip_frame=1,
    history_len=2,
    prediction_offset=5,
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

    norm_stats = ImageDataset.get_norm_stats(dataset_dirs, num_episodes_list)

    train_ratio = 0.8
    val_ratio = 0.1

    shuffled_indices = np.random.permutation(all_episode_indices)
    train_split = int(train_ratio * len(all_episode_indices))
    val_split = int((train_ratio + val_ratio) * len(all_episode_indices))
    
    train_indices = shuffled_indices[:train_split]
    val_indices = shuffled_indices[train_split:val_split]
    pretest_indices = shuffled_indices[val_split:]

    train_datasets = [
        ImageDataset(
            [idx for d, idx in train_indices if d == dataset_dir],
            dataset_dir,
            camera_names,
            norm_stats,
            history_skip_frame,
            history_len,
            prediction_offset,
            max_len,
            policy_class=policy_class,
        )
        for dataset_dir in dataset_dirs
    ]
    val_datasets = [
        ImageDataset(
            [idx for d, idx in val_indices if d == dataset_dir], 
            dataset_dir, 
            camera_names, 
            norm_stats,
            history_skip_frame,
            history_len,
            prediction_offset,
            max_len, 
            policy_class=policy_class
        ) 
        for dataset_dir in dataset_dirs
    ]
    
    pretest_datasets = [
        ImageDataset(
            [idx for d, idx in pretest_indices if d == dataset_dir],
            dataset_dir,
            camera_names,
            norm_stats,
            history_skip_frame,
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
        batch_size=batch_size_train,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        prefetch_factor=1,
    )
    return train_dataloader, norm_stats, val_dataloader, pretest_dataloader, test_dataloader


"""
Test the Dataset class.

Example usage:
$ python new_image_dataset.py --dataset_dir /mnt/d/kit/ALR/dataset/ttp_compressed/
"""
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--dataset_dir", type=str, required=True, help="Path to the dataset directory"
#     )
#     args = parser.parse_args()

#     camera_names = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
#     history_len = 3
#     prediction_offset = 8
#     num_episodes = 50  # Just to sample from the first 50 episodes for testing
#     history_skip_frame = 10
#     norm_stats = ImageDataset.get_norm_stats([args.dataset_dir], [num_episodes])
#     max_len = history_len + prediction_offset + 1
    
#     dataset = ImageDataset(
#         list(range(num_episodes)),
#         args.dataset_dir,
#         camera_names,
#         norm_stats,
#         history_skip_frame,
#         history_len,
#         prediction_offset,
#         max_len,
#         policy_class="Diffusion",
#     )
    
#     idx = np.random.randint(0, len(dataset))
#     image_sequence, action_data, is_pad = dataset[idx]

#     print(f"Sampled episode index: {idx}")

#     output_dir = os.path.join(dataset.dataset_dir,"plot")
#     os.makedirs(output_dir, exist_ok=True)
    
#     for t in tqdm(range(history_len+prediction_offset+1)):
#         plt.figure(figsize=(10, 5))
#         for cam_idx, cam_name in enumerate(camera_names):
#             plt.subplot(1, len(camera_names), cam_idx + 1)
#             img_rgb = cv2.cvtColor(
#                 image_sequence[t, cam_idx].permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB
#             )
#             plt.imshow(img_rgb)
#             plt.title(f"{cam_name} at timestep {t}")
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f"image_sequence_timestep_{t}.png"))
#         print(f"Saved image_sequence_timestep_{t}.png")
#         plt.close()
