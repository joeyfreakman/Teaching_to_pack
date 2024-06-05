#!/usr/bin/env python3
DATA_DIR = '/root/Teaching-to-pack/environment/dataset/data'
Data_config ={
    'teachingtopack':{
        'dataset_dir': DATA_DIR + '/teachingtopack',
        'num_episodes': 50,
        'episode_len': 1000,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
}