import os
from torch.utils.data import Dataset
import abc
from dataset_config import Data_config
from scripts.data_pruning import random_crop

class Highdim_dataset(Dataset, abc.ABC):
    def __init__(self,
                  episode_ids,
        dataset_dir,
        camera_names= Data_config['teachingtopack']['camera_names'],
        history_len: int = 2,
        prediction_offset: int = 16,
        history_skip_frame: int = 1,
        random_crop =False,
        device: str = "cpu"
                  ):
        self.episode_ids = episode_ids if len(episode_ids) > 0 else [0]
        self.dataset_dir = dataset_dir
        self.camera_names= camera_names
        self.history_len = history_len
        self.prediction_offset = prediction_offset
        self.history_skip_frame = history_skip_frame
        self.random_crop = random_crop
        self.device = device

    
    def get_seq_length(self, idx):
        raise NotImplementedError
    
    
    def get_all_actions(self):
        raise NotImplementedError
        
    
    def get_all_observations(self):
        raise NotImplementedError


# class Lowdim_Dataset(Dataset, abc.ABC):
#     def __init__(self,
#                  data_directory: os.PathLike,
#                  device="cpu",
#                  obs_dim: int = 514,
#                  action_dim: int = 2,
#                  max_len_data: int = 256,
#                  window_size: int = 1
#                  ):
#         self.data_directory = data_directory
#         self.device = device
#         self.max_len_data = max_len_data
#         self.action_dim = action_dim
#         self.obs_dim = obs_dim
#         self.window_size = window_size

#     @abc.abstractmethod
#     def get_seq_length(self, idx):
#         raise NotImplementedError

#     @abc.abstractmethod
#     def get_all_actions(self):
#         raise NotImplementedError

#     @abc.abstractmethod
#     def get_state_observations(self):
#         raise NotImplementedError
    