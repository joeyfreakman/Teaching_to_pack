import numpy as np
import torch
import os
import h5py
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from src.model.util import DAggerSampler
import argparse
import time
from tqdm import tqdm
import tikzplotlib
import matplotlib.pyplot as plt


class CurrImageDataset(Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        max_len=None,
        policy_class=None,
    ):
        super().__init__()
        self.episode_ids = episode_ids if len(episode_ids) > 0 else [0]
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.max_len = max_len
        self.policy_class = policy_class
        self.episode_starts = [0]
        self.total_len = 0
        self.transformations = None
        for episode_id in self.episode_ids:
            with h5py.File(os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5"), "r") as f:
                episode_len = len(f["/action"])
                self.total_len += episode_len
                self.episode_starts.append(self.total_len)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        episode_idx = 0
        while index >= self.episode_starts[episode_idx + 1]:
            episode_idx += 1
        local_index = index - self.episode_starts[episode_idx]

        episode_id = self.episode_ids[episode_idx]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        
        with h5py.File(dataset_path, "r") as root:
            compressed = root.attrs.get("compress", False)
            episode_len = len(root["/action"])
            start_ts = local_index
            end_ts = min(start_ts + self.max_len, episode_len)

            # Load and process images
            image_dict = {cam: root[f"/observations/images/{cam}"][start_ts] for cam in self.camera_names}
            if compressed:
                image_dict = {cam: cv2.imdecode(img, 1) for cam, img in image_dict.items()}

            # Convert images to RGB and stack them
            all_cam_images = np.stack([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image_dict.values()])
            image_data = torch.from_numpy(all_cam_images).permute(0, 3, 1, 2).float() / 255.0

            # Process actions and padding
            action = root["/action"][start_ts:end_ts]
            padded_action = np.zeros((self.max_len,) + action.shape[1:], dtype=np.float32)
            padded_action[:len(action)] = action
            is_pad = torch.zeros(self.max_len, dtype=torch.bool)
            is_pad[len(action):] = True

            action_data = 2 * (torch.from_numpy(padded_action) - self.norm_stats["action_min"]) / \
                        (self.norm_stats["action_max"] - self.norm_stats["action_min"]) - 1

            # Initialize transformations if needed
            if self.transformations is None:
                self.initialize_transformations(image_data.shape[2:])

            for transform in self.transformations:
                image_data = transform(image_data).float()

            return image_data, action_data, is_pad

    def initialize_transformations(self, original_size):
        ratio = 0.95
        self.transformations = [
            transforms.RandomCrop([int(original_size[0] * ratio), int(original_size[1] * ratio)]),
            transforms.Resize(original_size, antialias=True)
        ]
        if self.policy_class == "Diffusion":
            self.transformations.extend([
                transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
            ])
    
    @staticmethod
    def get_norm_stats(dataset_dirs, num_episodes_list):
        all_action_data = []

        for dataset_dir, num_episodes in zip(dataset_dirs, num_episodes_list):
            for episode_idx in range(num_episodes):
                dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
                with h5py.File(dataset_path, "r") as root:
                    action = root["/action"][()]
                all_action_data.append(torch.from_numpy(action))

        all_action_data = torch.cat(all_action_data, dim=0)

        action_mean = all_action_data.mean(dim=[0]).float()
        action_std = all_action_data.std(dim=[0]).float()
        action_std = torch.clip(action_std, 1e-2, np.inf)

        action_min = all_action_data.min(dim=0).values.float()
        action_max = all_action_data.max(dim=0).values.float()
        eps = 0.0001

        stats = {
            "action_mean": action_mean.numpy(),
            "action_std": action_std.numpy(),
            "action_min": action_min.numpy() - eps,
            "action_max": action_max.numpy() + eps,
        }

        return stats

### Merge multiple datasets
def load_merged_data(
    dataset_dirs,
    num_episodes_list,
    camera_names,
    batch_size_train,
    max_len=None,
    dagger_ratio=None,
    policy_class=None,
):
    assert len(dataset_dirs) == len(num_episodes_list), "Length of dataset_dirs and num_episodes_list must be the same."
    if dagger_ratio is not None:
        assert 0 <= dagger_ratio <= 1, "dagger_ratio must be between 0 and 1."

    all_episode_indices = [(dir, i) for dir, num_episodes in zip(dataset_dirs, num_episodes_list) for i in range(num_episodes)]
    last_dataset_indices = [(dataset_dirs[-1], i) for i in range(num_episodes_list[-1])]

    norm_stats = CurrImageDataset.get_norm_stats(dataset_dirs, num_episodes_list)

    shuffled_indices = np.random.permutation(all_episode_indices)
    train_split, val_split = int(0.8 * len(all_episode_indices)), int(0.9 * len(all_episode_indices))
    
    splits = [shuffled_indices[:train_split], shuffled_indices[train_split:val_split], shuffled_indices[val_split:]]
    
    def create_datasets(indices):
        return [CurrImageDataset(
            [idx for d, idx in indices if d == dataset_dir],
            dataset_dir,
            camera_names,
            norm_stats,
            max_len,
            policy_class=policy_class,
        ) for dataset_dir in dataset_dirs]

    train_datasets, val_datasets, pretest_datasets = map(create_datasets, splits)
    
    for dataset in train_datasets + val_datasets + pretest_datasets:
        if len(dataset) == 0:
            print(f"Warning: Empty dataset found in {dataset.dataset_dir}")

    merged_datasets = [ConcatDataset(datasets) for datasets in [train_datasets, val_datasets, pretest_datasets, train_datasets + val_datasets + pretest_datasets]]

    dataloader_params = {
        'pin_memory': True,
        'num_workers': 6,
        'prefetch_factor': 8,
        'persistent_workers': True
    }

    if dagger_ratio is not None:
        dataset_sizes = dict(zip(dataset_dirs, num_episodes_list))
        dagger_sampler = DAggerSampler(all_episode_indices, last_dataset_indices, batch_size_train, dagger_ratio, dataset_sizes)
        train_dataloader = DataLoader(merged_datasets[0], batch_sampler=dagger_sampler, **dataloader_params)
        val_dataloader = DataLoader(merged_datasets[1], batch_size=batch_size_train, **dataloader_params)
    else:
        train_dataloader = DataLoader(merged_datasets[0], batch_size=batch_size_train, shuffle=True, **dataloader_params)
        val_dataloader = DataLoader(merged_datasets[1], batch_size=batch_size_train, shuffle=True, **dataloader_params)

    pretest_dataloader = DataLoader(merged_datasets[2], batch_size=batch_size_train, shuffle=False, **{**dataloader_params, 'num_workers': 2, 'prefetch_factor': 8})
    test_dataloader = DataLoader(merged_datasets[3], batch_size=batch_size_train, shuffle=True, **{**dataloader_params, 'num_workers': 2, 'prefetch_factor': 8})
 
    return train_dataloader, norm_stats, val_dataloader, pretest_dataloader, test_dataloader

"""
Test the Dataset class.

Example usage:
$ python /root/Teaching_to_pack/environment/dataset/curr_image_dataset.py --dataset_dir /mnt/d/kit/ALR/dataset/ttp_compressed/
"""
if __name__ == "__main__":
    t_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to the dataset directory"
    )
    args = parser.parse_args()

    camera_names = ["cam_high","cam_left_wrist", "cam_low",  "cam_right_wrist"]
    num_episodes = 50 # Just to sample from the first 50 episodes for testing
    norm_stats = CurrImageDataset.get_norm_stats([args.dataset_dir], [num_episodes])
    max_len = 40
    dataset = CurrImageDataset(
        list(range(num_episodes)),
        args.dataset_dir,
        camera_names,
        norm_stats,
        max_len,
        policy_class="Diffusion",
    )
    stats = CurrImageDataset.get_norm_stats([args.dataset_dir], [num_episodes])
    dataset_name = 'episode_0'
    save_dir = os.path.join(args.dataset_dir, "test")
    post_process = (
        lambda a: ((a + 1) / 2) * (stats["action_max"] - stats["action_min"])
        + stats["action_min"]
    )
    # idx=0
    # image_data, action_data, is_pad = dataset[idx]
    # true_action = post_process(action_data.numpy())

    # _, _, original_action, _ = load_hdf5(args.dataset_dir, dataset_name)

    # # 确保true_action长度与原始action相同
    # original_action = original_action[idx:len(true_action)+idx]

    # visualize_joints(original_action, true_action, plot_path=save_dir)
    # print(f"Testing completed. Results saved in: {save_dir}")

    # tru_traj = []
    # for i in range(0, 750,16):  # 每16步取一次数据
    #     image_data, action_data, is_pad = dataset[i]  # 使用整除来获取正确的索引
    #     tru_traj.extend(action_data)
    # tru_traj = post_process(np.array(tru_traj))
    # # 确保轨迹长度为750
    # tru_traj = np.array(tru_traj[:750])


    # # 加载原始数据进行比较
    # _, _, original_action, _ = load_hdf5(args.dataset_dir, dataset_name)

    # # 确保original_action长度与tru_traj相同
    # original_action = original_action[:750]

    
    # visualize_joints(original_action, tru_traj, plot_path=save_dir)
    # print(f"Testing completed. Results saved in: {save_dir}")

    # tru_traj = []
    # _, _, _, original_action, _ = load_hdf5(args.dataset_dir, dataset_name)
    # test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # for idx, data in enumerate(test_dataloader):

    #     image_data, action_data, is_pad = data
    #     true_action = post_process(action_data[0])

    #     while idx <750:
    #         if  idx ==1 or idx % 16 == 0:

    #             tru_traj.extend(true_action)

    #         if len(tru_traj)>=750:
    #             true_traj = np.array(tru_traj[:750])
    #             visualize_joints(original_action, true_traj, plot_path=save_dir)
    #             print(f"Testing completed. Results saved in: {save_dir}")
    #             break


    
    
            
            

    # idx = 48    
    idx = np.random.randint(0, len(dataset))
    print(f"dataset_len: {len(dataset)}")
    image_sequence, action_data, is_pad = dataset[idx]
    print(f"Sampled dataset index: {idx}")
    print(f"Image sequence shape: {image_sequence.shape}")
    # print(f"Action data shape: {action_data.shape}")
    # print(f'action_data: {action_data}')    
    # print(f"Is pad : {is_pad}")
    output_dir = os.path.join(dataset.dataset_dir,"plot")
    os.makedirs(output_dir, exist_ok=True)
    
    for t in tqdm(range(1)):
        plt.figure(figsize=(10, 5))
        for cam_idx, cam_name in enumerate(camera_names):
            plt.subplot(1, len(camera_names), cam_idx + 1)
            img_rgb = image_sequence[cam_idx].permute(1, 2, 0).numpy().astype(np.float32)  # Convert to float32
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            # plt.imshow(img_rgb)
            plt.title(f"{cam_name} at timestep {t}")
    
        plt.tight_layout()
        tikzplotlib.save(os.path.join(output_dir, f"image_sequence_timestep_{t}.tex"))
        print(f"Saved image_sequence_timestep_{t}.png")
        plt.close()

    print(f"Time taken: {time.time() - t_start:.5f}s")