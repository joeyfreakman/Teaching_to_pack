import os
import cv2
import h5py
import numpy as np
from functools import cached_property
from typing import Union, Dict
import math
import h5py_cache
import time
class ReplayBuffer:
    """
    HDF5-based temporal datastructure.
    """
    def __init__(self, root: Union[h5py.Group, Dict[str, dict], h5py.File]):
        """
        load data from hdf5 file on the disk
        """
        assert 'observations' in root
        assert 'action' in root
        
        self.root = root
        self.compressed = self.root.attrs.get('compress', False)
        self.camera_names = list(self.root['observations']['images'].keys())
        for key, value in root['observations']['images'].items():
            if self.compressed:
                assert len(value.shape) == 2
            else:
                assert len(value.shape) == 4  # time dimension + (480, 640, 3)

    # ============= create constructors ===============
    @classmethod
    def create_from_path(cls, hdf5_path, mode='r', **kwargs):
        group = h5py.File(os.path.expanduser(hdf5_path), mode)
        buffer = cls(root=group)
        return buffer

    # ============= save methods ===============
    def save_to_store(self, store, **kwargs):
        root = h5py.File(store, 'w')
        obs_group = root.create_group('observations')
        images_group = obs_group.create_group('images')
        for key, value in self.root['observations']['images'].items():
            images_group.create_dataset(name=key, data=value, maxshape=(None, 480, 640, 3), dtype='uint8', chunks=self.get_optimal_chunks(value.shape, 'uint8'))
        obs_group.create_dataset(name='qpos', data=self.root['observations']['qpos'], maxshape=(None, 14), dtype='float64', chunks=self.get_optimal_chunks(self.root['observations']['qpos'].shape, 'float64'))
        obs_group.create_dataset(name='qvel', data=self.root['observations']['qvel'], maxshape=(None, 14), dtype='float64', chunks=self.get_optimal_chunks(self.root['observations']['qvel'].shape, 'float64'))
        root.create_dataset(name='action', data=self.root['action'], maxshape=(None, 14), dtype='float64', chunks=self.get_optimal_chunks(self.root['action'].shape, 'float64'))
        return store

    def save_to_path(self, hdf5_path, **kwargs):
        store = hdf5_path
        return self.save_to_store(store, **kwargs)
    
    # ============= properties ===============
    def decompress_image(self, img_data):
        if self.compressed:
            return np.array([cv2.imdecode(img, 1) for img in img_data])
        return img_data 

    @cached_property
    def data(self):
        return self.root

    @property
    def observations(self):
        return self.root['observations']

    @property
    def actions(self):
        return self.root['action']

    @property
    def n_steps(self):
        return self.actions.shape[0]

    @property
    def __repr__(self) -> str:
        return f"<ReplayBuffer with {self.n_episodes} episodes and {self.n_steps} steps>"

    @staticmethod
    def get_optimal_chunks(shape, dtype, target_chunk_bytes=2e6, max_chunk_length=None):
        """
        Common shapes
        T,D
        T,N,D
        T,H,W,C
        T,N,H,W,C
        """
        itemsize = np.dtype(dtype).itemsize
        # reversed
        rshape = list(shape[::-1])
        if max_chunk_length is not None:
            rshape[-1] = int(max_chunk_length)
        split_idx = len(shape) - 1
        for i in range(len(shape) - 1):
            this_chunk_bytes = itemsize * np.prod(rshape[:i])
            next_chunk_bytes = itemsize * np.prod(rshape[:i + 1])
            if this_chunk_bytes <= target_chunk_bytes \
                    and next_chunk_bytes > target_chunk_bytes:
                split_idx = i

        rchunks = rshape[:split_idx]
        item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
        this_max_chunk_length = rshape[split_idx]
        next_chunk_length = min(this_max_chunk_length, math.ceil(
            target_chunk_bytes / item_chunk_bytes))
        rchunks.append(next_chunk_length)
        len_diff = len(shape) - len(rchunks)
        rchunks.extend([1] * len_diff)
        chunks = tuple(rchunks[::-1])
        return chunks

    def get_step(self, idx):
        result = {
            'observations': {
                'images': {},
                'qpos': None,
                'qvel': None
            },
            'action': None
        }
        
        for cam_name in self.camera_names:
            img_data = self.observations['images'][cam_name][idx]
            result['observations']['images'][cam_name] = self.decompress_image(img_data)
        
        result['observations']['qpos'] = self.observations['qpos'][idx]
        result['observations']['qvel'] = self.observations['qvel'][idx]
        result['action'] = self.actions[idx]
        
        return result

    def get_steps_slice(self, start, stop, step=None):
        _slice = slice(start, stop, step)

        result = {
            'observations': {
                'images': {},
                'qpos': None,
                'qvel': None
            },
            'action': None
        }
        
        for cam_name in self.camera_names:
            img_data = self.observations['images'][cam_name][_slice]
            result['observations']['images'][cam_name] = self.decompress_image(img_data)
        
        result['observations']['qpos'] = self.observations['qpos'][_slice]
        result['observations']['qvel'] = self.observations['qvel'][_slice]
        result['action'] = self.actions[_slice]
        
        return result


    def optimize_chunks_and_cache(self, hdf5_path, cache_size=15 * 1024 * 1024, **kwargs):
        """
        Optimize chunks for the given hdf5 file and save it in a cache.
        """
    # Read the original data
        with h5py.File(os.path.expanduser(hdf5_path), 'r',rdcc_nbytes= cache_size) as f:
            data = {
                'observations': {
                    'images': {cam: self.decompress_image(f[f'observations/images/{cam}'][:]) for cam in self.camera_names},
                    'qpos': f['observations/qpos'][:],
                    'qvel': f['observations/qvel'][:]
                },
                'action': f['action'][:]
            }

        # Create a new HDF5 file with optimized chunks
        optimized_path = hdf5_path.replace('.hdf5', '_optimized.hdf5')
        with h5py.File(optimized_path, 'w', rdcc_nbytes= cache_size) as f:
            f.attrs['compress'] = False
            f.attrs['is_sim']= False
            obs_group = f.create_group('observations')
            images_group = obs_group.create_group('images')
            for key, value in data['observations']['images'].items():
                images_group.create_dataset(name=key, data=value, maxshape=(None, 480, 640, 3), dtype='uint8',compression='lzf',chunks=(1, 480, 640, 3),shuffle=True)
            obs_group.create_dataset(name='qpos', data=data['observations']['qpos'], maxshape=(None, 14), dtype='float64', chunks=self.get_optimal_chunks(data['observations']['qpos'].shape, 'float64'))
            obs_group.create_dataset(name='qvel', data=data['observations']['qvel'], maxshape=(None, 14), dtype='float64', chunks=self.get_optimal_chunks(data['observations']['qvel'].shape, 'float64'))
            f.create_dataset(name='action', data=data['action'], maxshape=(None, 14), dtype='float64', chunks=self.get_optimal_chunks(data['action'].shape, 'float64'))
        
        # Load the optimized HDF5 file as the current root
        # self.root = h5py_cache.File(optimized_path, 'r', chunk_cache_mem_size=cache_size)
    def check_chunks(self,hdf5_path):
        with h5py.File(hdf5_path,'r') as root:
            for key in root['observations']['images']:
                chunks = root['observations']['images'][key].chunks
                dtype_size = root['observations']['images'][key].dtype.itemsize
                chunk_num_elements = np.prod(chunks)
                chunk_size = chunk_num_elements * dtype_size
                print(f'{key} chunks: {chunks}', f'chunk size: {chunk_size} bytes')
            chunks_pos = root['observations']['qpos'].chunks
            chunks_action = root['action'].chunks
            print(f'action chunks: {chunks_action}, qpos chunks: {chunks_pos}')
            Compress = root.attrs.get('compress', False)
            print(f'Compress: {Compress}')
            

    
    def data_processing(self, max_timesteps, camera_names, dataset_path):
        """
        Process and save data in HDF5 format.
        
        For each timestep:
        observations
        - images
            - cam_high          (480, 640, 3) 'uint8'
            - cam_low           (480, 640, 3) 'uint8'
            - cam_left_wrist    (480, 640, 3) 'uint8'
            - cam_right_wrist   (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'
        - effort                (14,)         'float64'
        action                  (14,)         'float64'
        """
        dataset_name = os.path.basename(dataset_path)
        with h5py.File(dataset_path, "r") as original_file:
        # 创建新的 chunked 文件
            new_dataset_path = dataset_path.replace('.hdf5', '_chunks.hdf5')
            with h5py.File(new_dataset_path, "w", rdcc_nbytes=10*1024*1024) as chunked_file:
                # 复制文件属性
                for key, value in original_file.attrs.items():
                    chunked_file.attrs[key] = value
                
                # 创建 observations 组
                obs = chunked_file.create_group("observations")
                
                # 复制并 chunk qpos, qvel, effort 数据
                for key in ['qpos', 'qvel', 'effort']:
                    if key in original_file["observations"]:
                        original_dataset = original_file["observations"][key]
                        chunked_dataset = obs.create_dataset(
                            key,
                            original_dataset.shape,
                            dtype=original_dataset.dtype,
                            chunks=(max_timesteps, 14)
                        )
                        chunked_dataset[:] = original_dataset[:]
                
                # 创建 images 组
                image = obs.create_group("images")
                
                # 复制并 chunk 图像数据
                for cam_name in camera_names:
                    original_dataset = original_file["observations"]["images"][cam_name]
                    chunk_size = (1, original_dataset.shape[1])  # 每个 chunk 包含一张图片
                    chunked_dataset = image.create_dataset(
                        cam_name,
                        original_dataset.shape,
                        dtype=original_dataset.dtype,
                        chunks=chunk_size
                    )
                    chunked_dataset[:] = original_dataset[:]
                
                # 复制并 chunk action 数据
                if "action" in original_file:
                    original_dataset = original_file["action"]
                    chunked_dataset = chunked_file.create_dataset(
                        "action",
                        original_dataset.shape,
                        dtype=original_dataset.dtype,
                        chunks=(max_timesteps, 14)
                    )
                    chunked_dataset[:] = original_dataset[:]
                
                # 复制 compress_len 数据（如果存在）
                if "compress_len" in original_file:
                    chunked_file.copy(original_file["compress_len"], "compress_len")

        print(f"{dataset_name} chunked saving completed")

        # print(f"{dataset_name} saving: {time.time() - t0:.1f} secs")
                
                    

if __name__ == "__main__":
    data_path = '/mnt/d/kit/ALR/dataset/ttp_compressed/'
    
    
    for idx in iter(range(5)):
        hdf5_path = os.path.join(data_path, f'episode_{idx}.hdf5')
        buffer = ReplayBuffer.create_from_path(hdf5_path)
        # buffer.optimize_chunks_and_cache(hdf5_path)
        buffer.data_processing(750, ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'], hdf5_path)
        new_hdf5_path = hdf5_path.replace('.hdf5', '_chunks.hdf5')
        buffer.check_chunks(new_hdf5_path)