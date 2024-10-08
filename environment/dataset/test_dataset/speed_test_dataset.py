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
import time

class SpeedtestDataset(torch.utils.data.Dataset):
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
        self.max_len = max_len
        self.policy_class = policy_class
        self.transformations = None
        self.history_len = history_len
        self.prediction_offset = prediction_offset
        self.history_skip_frame = history_skip_frame
        self.episode_starts = [0]
        self.total_len = 0
        for episode_id in self.episode_ids:
            with h5py.File(os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5"), "r") as f:
                episode_len = len(f["/action"])
                self.total_len += episode_len
                self.episode_starts.append(self.total_len)
        if self.policy_class == "Diffusion":
                self.transformations=transforms.Compose(
                        [
                            transforms.RandomRotation(degrees=[-2.0, 2.0], expand=False),
                            transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
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
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")

        with h5py.File(dataset_path, "r") as root:
            compressed = root.attrs.get("compress", False)
            total_timesteps = root["/action"].shape[0]
            valid_range = total_timesteps - self.prediction_offset - self.history_len * self.history_skip_frame
            curr_ts = self.history_len * self.history_skip_frame + (local_index % valid_range)
            start_ts = curr_ts - self.history_len * self.history_skip_frame
            target_ts = curr_ts + self.prediction_offset

            def load_image(t):
                images = {cam_name: root[f"/observations/images/{cam_name}"][t] for cam_name in self.camera_names}
                if compressed:
                    images = {cam_name: cv2.imdecode(img, 1) for cam_name, img in images.items()}
                return np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images.values()], axis=0)

            image_sequence = [load_image(t) for t in range(start_ts, curr_ts +1, self.history_skip_frame)]
            action_sequence = root["/action"][start_ts:target_ts+1]
            

            image_data = torch.tensor(np.stack(image_sequence, axis=0), dtype=torch.float32).permute(0, 1, 4, 2, 3) / 255.0
            
            action_data = torch.from_numpy(np.array(action_sequence)).float()

            action_len = len(action_sequence)
            if action_len < self.max_len:
                padded_action = torch.zeros((self.max_len, action_data.shape[1]), dtype=torch.float32)
                padded_action[:action_len] = action_data
                padded_action[action_len:] = action_data[-1]
                action_data = padded_action

            is_pad = torch.zeros(self.max_len, dtype=torch.bool)
            is_pad[action_len:] = True

            if self.transformations:
                image_data = torch.stack([self.transformations(img) for img in image_data])

            action_data = ((action_data - self.norm_stats["action_min"]) / (self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
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
    history_skip_frame=8,
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

    norm_stats = SpeedtestDataset.get_norm_stats(dataset_dirs, num_episodes_list)

    train_ratio = 0.8
    val_ratio = 0.1

    shuffled_indices = np.random.permutation(all_episode_indices)
    train_split = int(train_ratio * len(all_episode_indices))
    val_split = int((train_ratio + val_ratio) * len(all_episode_indices))
    
    train_indices = shuffled_indices[:train_split]
    val_indices = shuffled_indices[train_split:val_split]
    pretest_indices = shuffled_indices[val_split:]

    train_datasets = [
        SpeedtestDataset(
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
        SpeedtestDataset(
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
        SpeedtestDataset(
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


"""et
Test the Dataset class.

Example usage:
$ python /root/Teaching_to_pack/environment/dataset/test_dataset/speed_test_dataset.py --dataset_dir /mnt/d/kit/ALR/dataset/ttp_compressed/
"""
if __name__ == "__main__":
    t0= time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to the dataset directory"
    )
    args = parser.parse_args()

    camera_names = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    history_len = 1
    prediction_offset = 8
    num_episodes = 50  # Just to sample from the first 50 episodes for testing
    history_skip_frame = 7
    norm_stats = SpeedtestDataset.get_norm_stats([args.dataset_dir], [num_episodes])
    max_len = history_skip_frame*history_len + prediction_offset + 1
    obs_len = history_len + 1
    dataset = SpeedtestDataset(
        list(range(num_episodes)),
        args.dataset_dir,
        camera_names,
        norm_stats,
        history_skip_frame,
        history_len,
        prediction_offset,
        max_len,
        policy_class="Diffusion",
    )
    
    idx = np.random.randint(0, 7500)
    print(f'len(dataset): {len(dataset)}')
    image_sequence, action_data, is_pad = dataset[idx]
    # print(f"image_sequence.shape: {image_sequence.shape}")
    print(f"Sampled episode index: {idx}")
    # print(f"action_data.shape: {action_data.shape}")
    # print(f"is_pad.shape: {is_pad.shape}")
    output_dir = os.path.join(dataset.dataset_dir,"plot")
    os.makedirs(output_dir, exist_ok=True)
    
    for t in tqdm(range(obs_len)):
        plt.figure(figsize=(10, 5))
        for cam_idx, cam_name in enumerate(camera_names):
            plt.subplot(1, len(camera_names), cam_idx + 1)
            img_rgb = cv2.cvtColor(
                image_sequence[t, cam_idx].permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB
            )
            plt.imshow(img_rgb)
            plt.title(f"{cam_name} at timestep {t}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"image_sequence_timestep_{t}.png"))
        print(f"Saved image_sequence_timestep_{t}.png")
        plt.close()
    t1 = time.time()
    print(f"Time taken: {t1-t0:.2f} seconds")