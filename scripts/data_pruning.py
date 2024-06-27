"""by deleting extra time_step data to simplify the dataset.
Save the new dataset to a new hdf5 file.
python script/data_pruning.py --data_dir ~/dataset/ttp --start_idx 0 --end_idx 65
"""
import os
import h5py
import numpy as np
import argparse
from tqdm import tqdm
import cv2
import random
import sys



class Imagereader:
    def __init__(self, data_dir, start_idx, end_idx):
        self.data_dir = data_dir
        self.start_idx = start_idx
        self.end_idx = end_idx


    def load_hdf5(self, dataset_dir, dataset_name):
        dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
        if not os.path.isfile(dataset_path):
            print(f'Dataset does not exist at \n{dataset_path}\n')
            exit()

        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            self.is_sim = is_sim
            compressed = root.attrs.get("compress", False)
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            effort = root['/observations/effort'][()]
            action = root['/action'][()]
            image_dict = dict()
            for cam_name in root[f'/observations/images/'].keys():
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
                # print(f'Loaded {cam_name} image with shape: {image_dict[cam_name].shape}')
            if compressed:
                compress_len = root["/compress_len"][()]

        if compressed:
            for cam_id, cam_name in enumerate(image_dict.keys()):
                padded_compressed_image_list = image_dict[cam_name]
                image_list = []
                for frame_id, padded_compressed_image in enumerate(
                    padded_compressed_image_list
                ):
                    image_len = int(compress_len[cam_id, frame_id])
                    compressed_image = padded_compressed_image
                    image = cv2.imdecode(compressed_image, 1)
                    image_list.append(image)
                    # print(f'Loaded {cam_name} image with shape: {image.shape}')
                image_dict[cam_name] = image_list
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
def decompress_image(compressed_image):
    return cv2.imdecode(compressed_image, 1)

def crop_resize(image, crop_h=240, crop_w=320, resize_h=480, resize_w=640, resize=True):
    """
    Helper function to crop the bottom middle (offset by 20 pixels) and resize
    """
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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
    # return cropped_imagemnt


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="image data processing")
    parser.add_argument("--dataset_dir", type=str, default='/mnt/d/kit/ALR/dataset/ttp_compressed', help="Directory containing the dataset")
    parser.add_argument("--episode_idx", type=int, required=False,default=10, help="Episode index")
    parser.add_argument("--camera_name", type=str, required=False,default='cam_high', help="Camera name")
    parser.add_argument("--pic_idx", type=int, required=False,default=1, help="Picture index")

    args = parser.parse_args()

    image_dir = os.path.join(args.dataset_dir, f'episode_{args.episode_idx}')
    image_path = os.path.join(image_dir, f'episode_{args.episode_idx}_{args.camera_name}')
    image_path = os.path.join(image_path, f'image_{args.pic_idx}.png')  

    if not os.path.isfile(image_path):
        print(f'Image file does not exist: {image_path}')
        exit()

    with open(image_path, 'rb') as f:
        compressed_image = f.read()

    compressed_image_np = np.frombuffer(compressed_image, np.uint8)
    decompressed_image = decompress_image(compressed_image_np)
    center_cropped_image = center_crop(decompressed_image, crop_percentage=0.95)
    crop_resized_image = crop_resize(decompressed_image, resize=False)
    random_cropped_image = random_crop(decompressed_image, crop_percentage=0.95)
    # Optionally save the cropped image
    # output_path_1 = os.path.join(image_dir, f'{args.camera_name}_cropped.jpg')
    # output_path_2 = os.path.join(image_dir, f'{args.camera_name}_random_cropped.jpg')
    # output_path_3 = os.path.join(image_dir, f'{args.camera_name}_crop_resized.jpg')
    # cv2.imwrite(output_path_1, center_cropped_image)
    # cv2.imwrite(output_path_2, random_cropped_image)
    # cv2.imwrite(output_path_3, crop_resized_image)
    # print(f'Saved cropped image to: {output_path_1}')
    # print(f'Saved random cropped image to: {output_path_2}')
    # print(f'Saved crop resized image to: {output_path_3}')
    # test=Imagereader(args.dataset_dir, 10, 11)
    # if test.load_hdf5(args.dataset_dir, f'episode_{args.episode_idx}'):
    #     print(test.is_sim)