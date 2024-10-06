import redis
import numpy as np
import torch
import os
import h5py
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from src.model.util import DAggerSampler
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.aloha.aloha_scripts.visualize_episodes import visualize_joints,load_hdf5
import json

class RedisImageDataset(Dataset):
    def __init__(
        self,
        episode_ids,
        redis_host,
        redis_port,
        redis_db,
        camera_names,
        norm_stats,
        history_len: int,
        prediction_offset: int,
        max_len=None,
        policy_class=None,
    ):
        self.episode_ids = episode_ids
        self.r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.history_len = history_len
        self.prediction_offset = prediction_offset
        self.max_len = max_len or (history_len + prediction_offset + 1)
        self.policy_class = policy_class
        self.transformations = None
        self.obs_horizon = history_len + 1
        
        self.episode_starts = [0]
        self.total_len = 0
        for episode_id in self.episode_ids:
            action_data = np.frombuffer(self.r.get(f'episode:{episode_id}:action'), dtype=np.float32)
            episode_len = len(action_data)
            self.total_len += episode_len
            self.episode_starts.append(self.total_len)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        episode_idx = 0
        while index >= self.episode_starts[episode_idx + 1]:
            episode_idx += 1
        local_index = index - self.episode_starts[episode_idx]

        episode_id = self.episode_ids[episode_idx]
        
        action_data = np.frombuffer(self.r.get(f'episode:{episode_id}:action'), dtype=np.float32).reshape(-1, 14)
        
        max_start = min(local_index, len(action_data) - self.obs_horizon)
        
        image_sequence = []
        for cam_name in self.camera_names:
            shape = json.loads(self.r.get(f'episode:{episode_id}:images:{cam_name}:shape'))
            cam_images = self.r.lrange(f'episode:{episode_id}:images:{cam_name}', max_start, max_start + self.obs_horizon - 1)
            cam_images = [np.frombuffer(img, dtype=np.uint8).reshape(shape[1:]) for img in cam_images]
            image_sequence.append(cam_images)
        image_sequence = np.stack(image_sequence, axis=1)
        
        action_sequence = action_data[max_start:max_start + self.max_len]
        
        image_data = torch.from_numpy(image_sequence).float()
        image_data = torch.einsum("t k h w c -> t k c h w", image_data)
        
        action_data = torch.from_numpy(action_sequence).float()
        action_len = len(action_sequence)
        if action_len < self.max_len:
            padded_action = torch.zeros((self.max_len, action_data.shape[1]), dtype=torch.float32)
            padded_action[:action_len] = action_data
            padded_action[action_len:] = action_data[-1]
            action_data = padded_action

        is_pad = torch.zeros(self.max_len, dtype=torch.bool)
        is_pad[action_len:] = True

        if self.transformations is None:
            self.setup_transformations(image_data.shape[3:])

        for transform in self.transformations:
            image_data = torch.stack([transform(img) for img in image_data])

        image_data = image_data / 255.0
        
        action_data = (action_data - self.norm_stats["action_min"]) / (self.norm_stats["action_max"] - self.norm_stats["action_min"]) * 2 - 1
        
        return image_data, action_data, is_pad

    def setup_transformations(self, original_size):
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
            self.transformations.extend([
                transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
            ])

    @staticmethod
    def write_hdf5_to_redis(hdf5_path, redis_host='localhost', redis_port=6379, redis_db=0):
        r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        
        with h5py.File(hdf5_path, 'r') as f:
            episode_id = hdf5_path.split('_')[-1].split('.')[0]
            
            # write action
            action_data = f['/action'][:]
            r.set(f'episode:{episode_id}:action', action_data.tobytes())
            
            # write images
            for cam_name in f['/observations/images'].keys():
                images = f[f'/observations/images/{cam_name}'][:]
                compressed = f.attrs.get('compress', False)
                
                if compressed:
                    compress_len = f['/compress_len'][:]
                    for i, img in enumerate(images):
                        img_len = int(compress_len[list(f['/observations/images'].keys()).index(cam_name), i])
                        compressed_img = img[:img_len]
                        img = cv2.imdecode(compressed_img, 1)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        r.rpush(f'episode:{episode_id}:images:{cam_name}', img.tobytes())
                else:
                    for img in images:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        r.rpush(f'episode:{episode_id}:images:{cam_name}', img.tobytes())
                
                # save images shape
                r.set(f'episode:{episode_id}:images:{cam_name}:shape', json.dumps(images.shape))

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