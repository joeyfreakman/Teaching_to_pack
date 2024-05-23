#!/usr/bin/env python3
import logging

import os
import glob
from torch.utils.data import Dataset
import cv2
import h5py
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

class dataprocesser:
    def __init__(
                self,dataset_dir: str, dataset_name: str, dataset_path: str,
                device="cpu",
                img_dim: int = 3,
                action_dim: int = 1,
                pose_dim: int = 1,
                vel_dim: int = 1,
                max_len_data: int = 256,
                window_size: int = 1,
                n_components: int = 50
                ):
        
        self.device = device
        self.max_len_data = max_len_data
        self.action_dim = action_dim
        self.img_dim = img_dim
        self.pose_dim = pose_dim
        self.vel_dim = vel_dim
        self.window_size = window_size
        self.n_components = n_components
        self.dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
        if not os.path.isfile(dataset_path):
            print(f'Dataset does not exist at \n{dataset_path}\n')
            exit()

    def read_hdf5(self):
        
        with h5py.File(self.dataset_path, 'r') as raw_data:
            qpos = raw_data['/observations/qpos'][()]
            qvel = raw_data['/observations/qvel'][()]
            effort = raw_data['/observations/effort'][()]
            action = raw_data['/action'][()]
            image_dict = dict()
            for cam_name in raw_data[f'/observations/images/'].keys():
                image_dict[cam_name] = raw_data[f'/observations/images/{cam_name}'][()]
        return qpos, qvel, effort, action, image_dict
    
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

    def get_all_actions(self):
        qpos, qvel, effort, action, image_dict = self.read_hdf5()
        actions = torch.tensor(action, dtype=torch.float32)
        return actions
    
    def get_all_images(self):
        qpos, qvel, effort, action, image_dict = self.read_hdf5()
        images = dict()
        for cam_name in image_dict.keys():
            images[cam_name] = torch.tensor(image_dict[cam_name], dtype=torch.uint8)
        return images
    
    # PCA to reduce unnecessary features
    def image_preprocess(self, image_dict):
        pca = PCA(n_components=self.n_components)
        reduced_images = []
        
        for cam_name, images in image_dict.items():
            num_images = images.shape[0]
            images_reshaped = images.reshape(num_images, -1)  # flatten the images--1 dimension
            reduced_images_cam = pca.fit_transform(images_reshaped)
            reduced_images.append(reduced_images_cam)
        
        # concatenate the reduced images
        all_reduced_images = np.concatenate(reduced_images, axis=1)
        
        return all_reduced_images


    # Get all states
    def get_all_states(self):
        qpos, qvel, effort, action, image_dict = self.read_hdf5()
        states = torch.tensor(np.concatenate((qpos, qvel, effort), axis=1), dtype=torch.float32)
        robot_init_state = states[0]
        robot_dest_state = states[-1]
        return states
    
    def sequence_length(self):
        qpos, qvel, effort, action, image_dict = self.read_hdf5()
        return len(action)
    
    def get_all_observations(self):
        qpos, qvel, effort, _, image_dict = self.read_hdf5()

        # Retrieve all images using get_all_images and then preprocess
        image_dict = self.get_all_images()  # Get all images from the dataset
        image_features = self.image_preprocess(image_dict)  # Preprocess images

        states = np.concatenate((qpos, qvel, effort), axis=1)
        observations = np.concatenate((states, image_features), axis=1)
        return torch.tensor(observations, dtype=torch.float32)

"""
The aim is to deal with the recorded data from the environment and concantenate the observations into a single tensor.

"""