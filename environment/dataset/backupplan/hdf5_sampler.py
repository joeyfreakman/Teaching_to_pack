from typing import Optional
import numpy as np
import numba
from environment.dataset.backupplan.hdf5_buffer import ReplayBuffer
import h5py

@numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray] = None,
        ):
        super().__init__()
        assert sequence_length >= 1
        if keys is None:
            keys = ['observations', 'action']
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        else:
            indices = np.zeros((0,4), dtype=np.int64)

        self.indices = indices 
        self.keys = list(keys)
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
    
    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        result = {
            'observations': {
                'images': {},
                'qpos': None,
                'qvel': None
            },
            'action': None
        }

        # Handle observations
        obs_data = self.replay_buffer['observations']
        
        # Handle images
        for cam_key in obs_data['images'].keys():
            input_arr = obs_data['images'][cam_key]
            if 'observations' in self.key_first_k:
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k['observations'], n_data)
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=np.nan, dtype=input_arr.dtype)
                sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
            else:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            
            data = self._pad_data(sample, sample_start_idx, sample_end_idx, input_arr.shape[1:], input_arr.dtype)
            result['observations']['images'][cam_key] = data

        # Handle qpos and qvel
        for key in ['qpos', 'qvel']:
            input_arr = obs_data[key]
            if 'observations' in self.key_first_k:
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k['observations'], n_data)
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=np.nan, dtype=input_arr.dtype)
                sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
            else:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            
            data = self._pad_data(sample, sample_start_idx, sample_end_idx, input_arr.shape[1:], input_arr.dtype)
            result['observations'][key] = data

        # Handle action
        action_arr = self.replay_buffer['action']
        if 'action' in self.key_first_k:
            n_data = buffer_end_idx - buffer_start_idx
            k_data = min(self.key_first_k['action'], n_data)
            sample = np.full((n_data,) + action_arr.shape[1:], 
                fill_value=np.nan, dtype=action_arr.dtype)
            sample[:k_data] = action_arr[buffer_start_idx:buffer_start_idx+k_data]
        else:
            sample = action_arr[buffer_start_idx:buffer_end_idx]
        
        data = self._pad_data(sample, sample_start_idx, sample_end_idx, action_arr.shape[1:], action_arr.dtype)
        result['action'] = data

        return result

    def _pad_data(self, sample, sample_start_idx, sample_end_idx, shape, dtype):
        if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
            data = np.zeros(
                shape=(self.sequence_length,) + shape,
                dtype=dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < self.sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
            return data
        return sample