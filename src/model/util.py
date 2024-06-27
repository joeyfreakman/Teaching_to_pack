import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Sampler
import random
import psutil
import time
import h5py_cache
import cv2
import os

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DAggerSampler(Sampler):
    def __init__(
        self, all_indices, last_dataset_indices, batch_size, dagger_ratio, dataset_sizes
    ):
        self.other_indices, self.last_dataset_indices = self._flatten_indices(
            all_indices, last_dataset_indices, dataset_sizes
        )
        print(
            f"Len of data from the last dataset: {len(self.last_dataset_indices)}, Len of data from other datasets: {len(self.other_indices)}"
        )
        self.batch_size = batch_size
        self.dagger_ratio = dagger_ratio
        self.num_batches = len(all_indices) // self.batch_size

    @staticmethod
    def _flatten_indices(all_indices, last_dataset_indices, dataset_sizes):
        flat_other_indices = []
        flat_last_dataset_indices = []
        cumulative_size = 0

        for dataset_dir, size in dataset_sizes.items():
            for idx in range(size):
                if (dataset_dir, idx) in last_dataset_indices:
                    flat_last_dataset_indices.append(cumulative_size + idx)
                elif (dataset_dir, idx) in all_indices:
                    flat_other_indices.append(cumulative_size + idx)
            cumulative_size += size

        return flat_other_indices, flat_last_dataset_indices

    def __iter__(self):
        num_samples_last = int(self.batch_size * self.dagger_ratio)
        num_samples_other = self.batch_size - num_samples_last

        for _ in range(self.num_batches):
            batch_indices = []

            if num_samples_last > 0 and self.last_dataset_indices:
                batch_indices.extend(
                    np.random.choice(
                        self.last_dataset_indices, num_samples_last, replace=True
                    )
                )

            if num_samples_other > 0 and self.other_indices:
                batch_indices.extend(
                    np.random.choice(
                        self.other_indices, num_samples_other, replace=True
                    )
                )

            np.random.shuffle(batch_indices)  # shuffle within each batch
            yield batch_indices

    def __len__(self):
        return self.num_batches
    
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach().cpu()
    return new_d

def is_multi_gpu_checkpoint(state_dict):
    """
    Check if the given state_dict is from a model trained on multiple GPUs using DataParallel.
    """
    # Check if any key starts with 'module.'
    return any(k.startswith("model.module.") for k in state_dict.keys())


def save_trajectory(
    dataset_path, timesteps, actions, camera_names, image_list=None
):
    # save trajectory
    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 640, 3) 'uint8'
        - cam_low           (480, 640, 3) 'uint8'
        - cam_left_wrist    (480, 640, 3) 'uint8'
        - cam_right_wrist   (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'
    - option                (1,)          'int'

    action                  (14,)         'float64'
    """

    data_dict = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/observations/effort": [],
        "/observations/option": [],
        "/action": [],
    }
    for cam_name in camera_names:
        data_dict[f"/observations/images/{cam_name}"] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict["/observations/qpos"].append(ts.observation["qpos"])
        data_dict["/observations/qvel"].append(ts.observation["qvel"])
        data_dict["/observations/effort"].append(ts.observation["effort"])
        option_expanded = np.expand_dims(np.array(ts.observation["option"]), axis=0)
        data_dict["/observations/option"].append(option_expanded)
        data_dict["/action"].append(action)
        for cam_name in camera_names:
            data_dict[f"/observations/images/{cam_name}"].append(
                ts.observation["images"][cam_name]
            )

    COMPRESS = True

    if COMPRESS:
        # JPEG compression
        t0 = time.time()
        encode_param = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            90,
        ]  # tried as low as 20, seems fine
        compressed_len = []
        for cam_name in camera_names:
            image_list_data = data_dict[f"/observations/images/{cam_name}"]
            compressed_list = []
            compressed_len.append([])
            for image in image_list_data:
                result, encoded_image = cv2.imencode(
                    ".jpg", image, encode_param
                )  # 0.02 sec # cv2.imdecode(encoded_image, 1)
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))
            data_dict[f"/observations/images/{cam_name}"] = compressed_list
        # print(f'compression: {time.time() - t0:.2f}s')

        # pad so it has same length
        t0 = time.time()
        compressed_len = np.array(compressed_len)
        padded_size = compressed_len.max()
        for cam_name in camera_names:
            compressed_image_list = data_dict[f"/observations/images/{cam_name}"]
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype="uint8")
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            data_dict[f"/observations/images/{cam_name}"] = padded_compressed_image_list
        # print(f'padding: {time.time() - t0:.2f}s')

    # HDF5
    t0 = time.time()
    max_timesteps = len(data_dict["/action"])
    with h5py_cache.File(
        dataset_path + ".hdf5", "w", chunk_cache_mem_size=1024**2 * 2
    ) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = COMPRESS
        obs = root.create_group("observations")
        image = obs.create_group("images")
        for cam_name in camera_names:
            if COMPRESS:
                _ = image.create_dataset(
                    cam_name,
                    (max_timesteps, padded_size),
                    dtype="uint8",
                    chunks=(1, padded_size),
                )
            else:
                _ = image.create_dataset(
                    cam_name,
                    (max_timesteps, 480, 640, 3),
                    dtype="uint8",
                    chunks=(1, 480, 640, 3),
                )
        _ = obs.create_dataset("qpos", (max_timesteps, 14))
        _ = obs.create_dataset("qvel", (max_timesteps, 14))
        _ = obs.create_dataset("effort", (max_timesteps, 14))
        _ = root.create_dataset("action", (max_timesteps, 14))

        for name, array in data_dict.items():
            root[name][...] = array

        if COMPRESS:
            _ = root.create_dataset("compress_len", (len(camera_names), max_timesteps))
            root["/compress_len"][...] = compressed_len
    
    # print(f'Saving: {time.time() - t0:.1f} secs')
    return ts, image_list


# Automatically kill the job if it’s going to exceed the memory limit.
def memory_monitor():
    while True:
        available_memory = psutil.virtual_memory().available / (
            1024**2
        )  # Available memory in MB
        if (
            available_memory < 1000
        ):  # MEMORY_BUFFER_MB: The amount of memory to ensure remains free
            print(
                f"Available memory is too low! {available_memory:.2f}MB left. Terminating..."
            )
            os._exit(1)
        time.sleep(5)

def get_auto_index(dataset_dir, dataset_name_prefix="", data_suffix="hdf5"):
    max_idx = 5000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx + 1):
        if not os.path.isfile(
            os.path.join(dataset_dir, f"{dataset_name_prefix}episode_{i}.{data_suffix}")
        ):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

def create_dataset_path(dataset_dir):
    episode_idx = get_auto_index(dataset_dir)
    dataset_name = f"episode_{episode_idx}"
    print(f"Dataset name: {dataset_name}")
    dataset_path = os.path.join(dataset_dir, dataset_name)
    return dataset_path, episode_idx
