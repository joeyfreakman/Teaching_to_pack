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
import IPython
from typing import Union
import numbers
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st
from src.model.Encoder.tensor_utils import NestedTensor

e = IPython.embed

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        if x.dim() == 3:
            B, T, D = x.shape
            x = x.view(B * T, D)
        
        emb = x[:, :, None] * emb[None, None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        if x.dim() == 3:
            emb = emb.view(B, T, -1)
        
        return emb


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
        )
        return pos

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor
        # mask = tensor_list.mask
        # assert mask is not None
        # not_mask = ~mask

        not_mask = torch.ones_like(x[0, [0]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class DAggerSampler(Sampler):
    def __init__(self, all_indices, last_dataset_indices, batch_size, dagger_ratio, dataset_sizes):
        self.other_indices, self.last_dataset_indices = self._flatten_indices(all_indices, last_dataset_indices, dataset_sizes)
        print(f"Len of data from the last dataset: {len(self.last_dataset_indices)}, Len of data from other datasets: {len(self.other_indices)}")
        self.batch_size = batch_size
        self.dagger_ratio = dagger_ratio
        self.num_batches = (len(self.other_indices) + len(self.last_dataset_indices)) // self.batch_size

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

        last_indices = np.array(self.last_dataset_indices)
        other_indices = np.array(self.other_indices)

        for _ in range(self.num_batches):
            batch_indices = []

            if num_samples_last > 0 and len(last_indices) > 0:
                batch_indices.extend(np.random.choice(last_indices, min(num_samples_last, len(last_indices)), replace=False))

            if num_samples_other > 0 and len(other_indices) > 0:
                batch_indices.extend(np.random.choice(other_indices, min(num_samples_other, len(other_indices)), replace=False))

            # 如果批次大小不足，从所有索引中随机填充
            if len(batch_indices) < self.batch_size:
                all_indices = np.concatenate([last_indices, other_indices])
                batch_indices.extend(np.random.choice(all_indices, self.batch_size - len(batch_indices), replace=False))

            np.random.shuffle(batch_indices)
            yield [int(idx) for idx in batch_indices]

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

def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding



def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    return (b * a.inv()).magnitude()

def pose_distance(start_pose, end_pose):
    start_pose = np.array(start_pose)
    end_pose = np.array(end_pose)
    start_pos = start_pose[:3]
    end_pos = end_pose[:3]
    start_rot = st.Rotation.from_rotvec(start_pose[3:])
    end_rot = st.Rotation.from_rotvec(end_pose[3:])
    pos_dist = np.linalg.norm(end_pos - start_pos)
    rot_dist = rotation_distance(start_rot, end_rot)
    return pos_dist, rot_dist

class PoseTrajectoryInterpolator:
    def __init__(self, times: np.ndarray, poses: np.ndarray):
        assert len(times) >= 1
        assert len(poses) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses)

        if len(times) == 1:
            # special treatment for single step interpolation
            self.single_step = True
            self._times = times
            self._poses = poses
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])

            pos = poses[:,:3]
            rot = st.Rotation.from_rotvec(poses[:,3:])

            self.pos_interp = si.interp1d(times, pos, 
                axis=0, assume_sorted=True)
            self.rot_interp = st.Slerp(times, rot)
    
    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        else:
            return self.pos_interp.x
    
    @property
    def poses(self) -> np.ndarray:
        if self.single_step:
            return self._poses
        else:
            n = len(self.times)
            poses = np.zeros((n, 6))
            poses[:,:3] = self.pos_interp.y
            poses[:,3:] = self.rot_interp(self.times).as_rotvec()
            return poses

    def trim(self, 
            start_t: float, end_t: float
            ) -> "PoseTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        # remove duplicates, Slerp requires strictly increasing x
        all_times = np.unique(all_times)
        # interpolate
        all_poses = self(all_times)
        return PoseTrajectoryInterpolator(times=all_times, poses=all_poses)
    
    def drive_to_waypoint(self, 
            pose, time, curr_time,
            max_pos_speed=np.inf, 
            max_rot_speed=np.inf
        ) -> "PoseTrajectoryInterpolator":
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        time = max(time, curr_time)
        
        curr_pose = self(curr_time)
        pos_dist, rot_dist = pose_distance(curr_pose, pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = time - curr_time
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        # insert new pose
        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        # create new interpolator
        final_interp = PoseTrajectoryInterpolator(times, poses)
        return final_interp

    def schedule_waypoint(self,
            pose, time, 
            max_pos_speed=np.inf, 
            max_rot_speed=np.inf,
            curr_time=None,
            last_waypoint_time=None
        ) -> "PoseTrajectoryInterpolator":
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        if last_waypoint_time is not None:
            assert curr_time is not None

        # trim current interpolator to between curr_time and last_waypoint_time
        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                # if insert time is earlier than current time
                # no effect should be done to the interpolator
                return self
            # now, curr_time < time
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                # if last_waypoint_time is earlier than start_time
                # use start_time
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        # end time should be the latest of all times except time
        # after this we can assume order (proven by zhenjia, due to the 2 min operations)

        # Constraints:
        # start_time <= end_time <= time (proven by zhenjia)
        # curr_time <= start_time (proven by zhenjia)
        # curr_time <= time (proven by zhenjia)
        
        # time can't change
        # last_waypoint_time can't change
        # curr_time can't change
        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)

        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)
        # after this, all waypoints in trimmed_interp is within start_time and end_time
        # and is earlier than time

        # determine speed
        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        pos_dist, rot_dist = pose_distance(pose, end_pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = end_time + duration

        # insert new pose
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        # create new interpolator
        final_interp = PoseTrajectoryInterpolator(times, poses)
        return final_interp


    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])
        
        pose = np.zeros((len(t), 6))
        if self.single_step:
            pose[:] = self._poses[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)

            pose = np.zeros((len(t), 6))
            pose[:,:3] = self.pos_interp(t)
            pose[:,3:] = self.rot_interp(t).as_rotvec()

        if is_single:
            pose = pose[0]
        return pose
