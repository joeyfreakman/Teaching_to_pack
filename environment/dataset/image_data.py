#!/usr/bin/env python3
import logging
from data import dataset_config
import os
import time
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import h5py
import random
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn


class dataprocesser(Dataset):
    def __init__(
                self,dataset_dir: str, dataset_name: str, dataset_path: str, 
                time_step: int=dataset_config.Data_config['teachingtopack']['episode_len'],
                device="cpu",
                img_dim: int = 3,
                action_dim: int = 1,
                max_len_data: int = 256,
                window_size: int = 1,
                n_components: int = 50
                ):
        self.time_step = time_step
        self.device = device
        self.max_len_data = max_len_data
        self.action_dim = action_dim
        self.img_dim = img_dim
        self.window_size = window_size
        self.n_components = n_components
        self.dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
        if not os.path.isfile(dataset_path):
            print(f'Dataset does not exist at \n{dataset_path}\n')
            exit()
        self.qpos, self.qvel, self.effort, self.actions, self.images = self.read_hdf5()
        
    def read_hdf5(self):
        
        with h5py.File(self.dataset_path, 'r') as raw_data:
            action = raw_data['/action'][()]
            image_dict = dict()
            for cam_name in raw_data[f'/observations/images/'].keys():
                image_dict[cam_name] = raw_data[f'/observations/images/{cam_name}'][()]
        return action, image_dict
    
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
    
    action                  (14,)         'float64'
    """

def crop_resize(image, crop_h=240, crop_w=320, resize_h=480, resize_w=640, resize=True):
    """
    Helper function to crop the bottom middle (offset by 20 pixels) and resize
    """
    h, w, _ = image.shape
    y1 = h - crop_h - 20  # Subtracting 20 to start 20 pixels above the bottom
    x1 = (w - crop_w) // 2
    cropped = image[y1 : y1 + crop_h, x1 : x1 + crop_w]
    return cv2.resize(cropped, (resize_w, resize_h)) if resize else cropped


def random_crop(image, crop_percentage=0.95):
    """
    Crop the given image by a random percentage without going out of boundary.
    """
    h, w, _ = image.shape
    new_h, new_w = int(h * crop_percentage), int(w * crop_percentage)
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    cropped_image = image[top : top + new_h, left : left + new_w, :]
    return cv2.resize(cropped_image, (w, h), interpolation=cv2.INTER_LINEAR)
    # return cropped_image


def center_crop(image, crop_percentage=0.95):
    """
    Crop the center of the given image by a specified percentage.
    """
    h, w, _ = image.shape
    new_h, new_w = int(h * crop_percentage), int(w * crop_percentage)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    cropped_image = image[top : top + new_h, left : left + new_w, :]
    return cv2.resize(cropped_image, (w, h), interpolation=cv2.INTER_LINEAR)
    # return cropped_image
