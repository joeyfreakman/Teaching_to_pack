"""by deleting extra time_step data to simplify the dataset.
Save the new dataset to a new hdf5 file.
python script/data_pruning.py --data_dir ~/dataset/ttp --start_idx 0 --end_idx 65
"""
import os
import h5py
import numpy as np
import argparse
from tqdm import tqdm

class DatasetPruner:
    def __init__(self, data_dir, start_idx, end_idx):
        self.data_dir = data_dir
        self.start_idx = start_idx
        self.end_idx = end_idx

    def prune_data(self):
        data_files = [f for f in os.listdir(self.data_dir) if f.endswith(".hdf5")]

        for data_file in data_files:
            file_path = os.path.join(self.data_dir, data_file)
            output_dir = os.path.join(self.data_dir, "pruned")
            os.makedirs(output_dir, exist_ok=True)
            output_file_path = os.path.join(output_dir, data_file)

            with h5py.File(file_path, "r") as f:
                # Read data
                qpos_data = f["/observations/qpos"][:]
                qvel_data = f["/observations/qvel"][:]
                effort_data = f["/observations/effort"][:]
                actions_data = f["/action"][:]
                images_data = {
                    cam_name: f[f"/observations/images/{cam_name}"][:]
                    for cam_name in f["/observations/images"].keys()
                }
                compress_len_data = f["/compress_len"][:] if "/compress_len" in f else None

                new_data = {
                    "qpos": [],
                    "qvel": [],
                    "effort": [],
                    "action": [],
                    "images": {cam_name: [] for cam_name in images_data.keys()},
                }

                