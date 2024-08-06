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
from environment.dataset.sampler import SequenceSampler,get_val_mask
from environment.dataset.replay_buffer import ReplayBuffer
import copy
from filelock import FileLock
import zarr

CROP_TOP = False  # hardcode

class SampleImageDataset(torch.utils.data.Dataset):
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
        use_cache=False,
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
        self.obs_horizon = history_len + 1
        # create replay buffer
        if use_cache:
            cache_zarr_path = dataset_dir + '.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = self._load_replay_buffer()
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        os.remove(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
            self.replay_buffer = replay_buffer
        else:
            self.replay_buffer = self._load_replay_buffer()

        val_mask = get_val_mask(self.replay_buffer.n_episodes,val_ratio=0.1,seed=42)
        train_mask = ~val_mask
        self.val_mask = val_mask

        # create sequence sampler
        self.sequence_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=history_len + prediction_offset + 1,
            pad_before=0,
            pad_after=0,
            key_first_k={'images':self.obs_horizon},
            episode_mask=train_mask,
        )

    def get_val_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.max_len,
            pad_before=0, 
            pad_after=0,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def _load_replay_buffer(self):
    # initialize ReplayBuffer
        root = ReplayBuffer.create_empty_zarr()
        
        # load data into replay buffer
        for episode_id in self.episode_ids:
            dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
            with h5py.File(dataset_path, "r") as f:
                episode_data = {}
                compressed = f.attrs.get("compress", False)
                
                for cam_name in self.camera_names:
                    img_data = f[f"/observations/images/{cam_name}"][:]
                    if compressed:
                        img_data = np.array([cv2.imdecode(img, 1) for img in img_data])
                    episode_data[f"images/{cam_name}"] = img_data
                
                action_data = f["/action"][:]
                episode_data["action"] = action_data
                
                root.add_episode(episode_data)
        
        return root
    
    def __len__(self):
        return len(self.sequence_sampler)

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)):
            raise TypeError(f"Index must be an integer, got {type(index)}")
        index = int(index)
        if index < 0 or index >= len(self.sequence_sampler):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.sequence_sampler)}")
        
        sequence_data = self.sequence_sampler.sample_sequence(index)
        
        image_sequence = []
        for cam_name in self.camera_names:
            cam_images = sequence_data['images'][f'{cam_name}'][:self.obs_horizon]
            image_sequence.append(cam_images)
        image_sequence = np.stack(image_sequence, axis=1)

        action_sequence = sequence_data['action']

        image_data = torch.from_numpy(image_sequence)
        image_data = torch.einsum("t k h w c -> t k c h w", image_data)

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
        print(f"Transformed image data range: {image_data.min()} to {image_data.max()}")
        image_data = image_data / 255.0
        print(f"Normalized image data range: {image_data.min()} to {image_data.max()}")
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
    




"""
Test the Dataset class.

Example usage:
$ python sample_image_dataset.py --dataset_dir /mnt/d/kit/ALR/dataset/ttp_compressed/
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to the dataset directory"
    )
    args = parser.parse_args()

    camera_names = ["cam_high","cam_left_wrist", "cam_low",  "cam_right_wrist"]
    history_len = 1
    prediction_offset = 14
    num_episodes = 50  # Just to sample from the first 50 episodes for testing
    norm_stats = SampleImageDataset.get_norm_stats([args.dataset_dir], [num_episodes])
    max_len = history_len + prediction_offset + 1
    obs_len = history_len + 1
    dataset = SampleImageDataset(
        list(range(num_episodes)),
        args.dataset_dir,
        camera_names,
        norm_stats,
        history_len,
        prediction_offset,
        max_len,
        policy_class="Diffusion",
        use_cache=True,
    )
    
    idx = np.random.randint(0, len(dataset))
    print(f"dataset_len: {len(dataset)}")
    image_sequence, action_data, is_pad = dataset[idx]

    print(f"Sampled episode index: {idx}")

    output_dir = os.path.join(dataset.dataset_dir,"plot")
    os.makedirs(output_dir, exist_ok=True)
    print(f"action_data: {action_data.shape}")
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