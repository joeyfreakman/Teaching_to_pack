import numpy as np
import torch
import os
import random
import h5py
import torch.utils.data
import cv2
from scripts.data_pruning import crop_resize
from torchvision import transforms
import torch.nn.functional as F
from resnet_model import get_resnet
from encoder import MultiImageObsEncoder
from torch.utils.data import DataLoader, ConcatDataset
from util import DAggerSampler
CROP_TOP = True  # hardcode
FILTER_MISTAKES = True  # Filter out mistakes from the dataset

class EncoderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        max_len=None,
        policy_class=None,
        encoder=MultiImageObsEncoder
    ):
        super().__init__()
        self.episode_ids = episode_ids if len(episode_ids) > 0 else [0]
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.max_len = max_len
        self.policy_class = policy_class
        self.transformations = None
        self.encoder = encoder

        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        max_len = self.max_len

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")

        with h5py.File(dataset_path, "r") as root:
            is_sim = root.attrs["sim"]
            self.is_sim = is_sim
            compressed = root.attrs.get("compress", False)
            original_action_shape = root["/action"].shape

            start_ts = np.random.choice(original_action_shape[0])
            end_ts = original_action_shape[0] - 1

            qpos = root["/observations/qpos"][start_ts]

            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]

            if compressed:
                for cam_name in image_dict.keys():
                    decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                    image_dict[cam_name] = np.array(decompressed_image)

                    if CROP_TOP and cam_name == "cam_high":
                        image_dict[cam_name] = crop_resize(image_dict[cam_name])

            for cam_name in image_dict.keys():
                image_dict[cam_name] = cv2.cvtColor(
                    image_dict[cam_name], cv2.COLOR_BGR2RGB
                )

            if is_sim:
                action = root["/action"][start_ts : end_ts + 1]
                action_len = end_ts - start_ts + 1
            else:
                action = root["/action"][max(0, start_ts - 1) : end_ts + 1]
                action_len = end_ts - max(0, start_ts - 1) + 1

            padded_action = np.zeros(
                (max_len,) + original_action_shape[1:], dtype=np.float32
            )
            padded_action[:action_len] = action
            is_pad = np.zeros(max_len)
            is_pad[action_len:] = 1

            all_cam_images = [image_dict[cam_name] for cam_name in self.camera_names]
            all_cam_images = np.stack(all_cam_images, axis=0)

            image_data = torch.from_numpy(all_cam_images)
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()

            image_data = torch.einsum("k h w c -> k c h w", image_data)

            if self.transformations is None:
                print("Initializing transformations")
                original_size = image_data.shape[2:]
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
                            transforms.RandomRotation(
                                degrees=[-5.0, 5.0], expand=False
                            ),
                            transforms.ColorJitter(
                                brightness=0.3, contrast=0.4, saturation=0.5 #change brightness, contrast, saturation of an image
                            ),
                        ]
                    )

            for transform in self.transformations:
                image_data = transform(image_data)

            image_data = image_data / 255.0
            # Use MultiImageObsEncoder to process image data
            obs_dict = {cam_name: image_data for cam_name in self.camera_names}
            encoded_images = self.encoder(obs_dict)

            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
                "qpos_std"
            ]

            # Combine encoded images and qpos to form the final observation
            observation = torch.cat([encoded_images, qpos_data], dim=-1)

            

            if self.policy_class == "Diffusion":
                action_data = (
                    (action_data - self.norm_stats["action_min"])
                    / (self.norm_stats["action_max"] - self.norm_stats["action_min"])
                ) * 2 - 1
            else:
                action_data = (
                    action_data - self.norm_stats["action_mean"]
                ) / self.norm_stats["action_std"]

            return observation, action_data, is_pad

def get_norm_stats(dataset_dirs, num_episodes_list):
    all_qpos_data = []
    all_action_data = []

    for dataset_dir, num_episodes in zip(dataset_dirs, num_episodes_list):
        for episode_idx in range(num_episodes):
            dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
            with h5py.File(dataset_path, "r") as root:
                qpos = root["/observations/qpos"][()]
                action = root["/action"][()]
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))

    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf)

    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()
    eps = 0.0001

    stats = {
        "action_mean": action_mean.numpy(),
        "action_std": action_std.numpy(),
        "action_min": action_min.numpy() - eps,
        "action_max": action_max.numpy() + eps,
        "qpos_mean": qpos_mean.numpy(),
        "qpos_std": qpos_std.numpy(),
        "example_qpos": all_qpos_data[-1].numpy(),
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
):
    assert len(dataset_dirs) == len(
        num_episodes_list
    ), "Length of dataset_dirs and num_episodes_list must be the same."
    if dagger_ratio is not None:
        assert 0 <= dagger_ratio <= 1, "dagger_ratio must be between 0 and 1."

    all_filtered_indices = []
    last_dataset_indices = []

    for i, (dataset_dir, num_episodes) in enumerate(
        zip(dataset_dirs, num_episodes_list)
    ):
        print(f"\nData from: {dataset_dir}\n")

        # Collect all episodes without filtering by command list
        filtered_indices = [(dataset_dir, i) for i in range(num_episodes)]

        if i == len(dataset_dirs) - 1:  # Last dataset
            last_dataset_indices.extend(filtered_indices)
        all_filtered_indices.extend(filtered_indices)

    print(f"Total number of episodes across datasets: {len(all_filtered_indices)}")

    # Obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dirs, num_episodes_list)

    # Construct dataset and dataloader for each dataset dir and merge them
    train_datasets = [
        EncoderDataset(
            [idx for d, idx in all_filtered_indices if d == dataset_dir],
            dataset_dir,
            camera_names,
            norm_stats,
            max_len,
            policy_class=policy_class,
        )
        for dataset_dir in dataset_dirs
    ]
    merged_train_dataset = ConcatDataset(train_datasets)

    if dagger_ratio is not None:
        dataset_sizes = {
            dataset_dir: num_episodes
            for dataset_dir, num_episodes in zip(dataset_dirs, num_episodes_list)
        }
        dagger_sampler = DAggerSampler(
            all_filtered_indices,
            last_dataset_indices,
            batch_size_train,
            dagger_ratio,
            dataset_sizes,
        )
        train_dataloader = DataLoader(
            merged_train_dataset,
            batch_sampler=dagger_sampler,
            pin_memory=True,
            num_workers=24,
            prefetch_factor=4,
            persistent_workers=True,
        )
    else:
        # Use default shuffling if dagger_ratio is not provided
        train_dataloader = DataLoader(
            merged_train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            pin_memory=True,
            num_workers=24,
            prefetch_factor=4,
            persistent_workers=True,
        )

    return train_dataloader, norm_stats, train_datasets[-1].is_sim