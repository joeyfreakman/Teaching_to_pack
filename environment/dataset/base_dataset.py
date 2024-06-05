import os
from torch.utils.data import Dataset
import abc

class Imagedataset(Dataset, abc.ABC):
    def __init__(self,
                  data_directory: os.PathLike, 
                  device="cpu", obs_dim: int = 4, 
                  action_dim: int = 2, max_len_data: int = 256, 
                  window_size: int = 1
                  ):
        self.data_directory = data_directory
        self.device = device
        self.max_len_data = max_len_data
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.window_size = window_size

    @abc.abstractmethod
    def get_seq_length(self, idx):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_all_actions(self):
        raise NotImplementedError
        
    @abc.abstractmethod
    def get_image_observations(self):
        raise NotImplementedError


class StateDataset(Dataset, abc.ABC):
    def __init__(self,
                 data_directory: os.PathLike,
                 device="cpu",
                 obs_dim: int = 2,
                 action_dim: int = 2,
                 max_len_data: int = 256,
                 window_size: int = 1
                 ):
        self.data_directory = data_directory
        self.device = device
        self.max_len_data = max_len_data
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.window_size = window_size

    @abc.abstractmethod
    def get_seq_length(self, idx):
        raise NotImplementedError

    @abc.abstractmethod
    def get_all_actions(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_state_observations(self):
        raise NotImplementedError
    