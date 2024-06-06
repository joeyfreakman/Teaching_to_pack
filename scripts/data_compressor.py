"""
by deleting extra time_step data, remarking transition markers to simplify the dataset.
Save the new dataset to a new hdf5 file.
$ python3 script/compress_data.py --dataset_dir ~/dataset/ttp

"""
import os
import h5py
import shutil
import numpy as np
import argparse
from tqdm import tqdm
import cv2
import random
class DatasetCompressor:
    def __init__(self, input_dataset_path, output_dataset_path):
        self.input_dataset_path = input_dataset_path
        self.output_dataset_path = output_dataset_path

    def compress_dataset(self):
        if not self._check_output_path():
            return

        with h5py.File(self.input_dataset_path, "r") as infile:
            with h5py.File(self.output_dataset_path, "w") as outfile:
                self._copy_non_image_data(infile, outfile)
                obs_group = infile["observations"]
                out_obs_group = outfile.create_group("observations")
                self._copy_non_image_observations(obs_group, out_obs_group)
                image_group = obs_group["images"]
                out_image_group = out_obs_group.create_group("images")
                compressed_lens = self._compress_and_store_images(image_group, out_image_group)
                outfile.create_dataset("compress_len", data=compressed_lens)

        print(f"Compressed dataset saved to {self.output_dataset_path}")

    def _check_output_path(self):
        if os.path.exists(self.output_dataset_path):
            print(f"The file {self.output_dataset_path} already exists. Exiting...")
            return False
        return True

    def _copy_non_image_data(self, infile, outfile):
        outfile.attrs["sim"] = infile.attrs["sim"]
        outfile.attrs["compress"] = True

        for key in infile.keys():
            if key != "observations":
                outfile.copy(infile[key], key)

    def _copy_non_image_observations(self, obs_group, out_obs_group):
        for key in obs_group.keys():
            if key != "images":
                out_obs_group.copy(obs_group[key], key)

    def _compress_and_store_images(self, image_group, out_image_group):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        compressed_lens = []

        for cam_name in image_group.keys():
            if "_depth" in cam_name:
                out_image_group.copy(image_group[cam_name], cam_name)
            else:
                images = image_group[cam_name]
                compressed_images, cam_compressed_lens = self._compress_images(images, encode_param)
                compressed_lens.append(cam_compressed_lens)

                max_len = max(len(img) for img in compressed_images)
                compressed_dataset = out_image_group.create_dataset(cam_name, (len(compressed_images), max_len), dtype="uint8")
                self._store_compressed_images(compressed_images, compressed_dataset)

        return np.array(compressed_lens)

    def _compress_images(self, images, encode_param):
        compressed_images = []
        cam_compressed_lens = []

        for image in images:
            result, encoded_image = cv2.imencode(".jpg", image, encode_param)
            compressed_images.append(encoded_image)
            cam_compressed_lens.append(len(encoded_image))

        return compressed_images, cam_compressed_lens

    def _store_compressed_images(self, compressed_images, compressed_dataset):
        for i, img in enumerate(compressed_images):
            compressed_dataset[i, :len(img)] = img

    def test(self,input_dataset_path,output_dataset_path):
        self.input_dataset_path = input_dataset_path
        self.output_dataset_path = output_dataset_path
        self.compress_dataset()

def get_auto_index(dataset_dir, dataset_name_prefix="", data_suffix="hdf5"):
    max_idx = 5000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx + 1):
        if not os.path.isfile(
            os.path.join(dataset_dir, f"{dataset_name_prefix}episode_{i}.{data_suffix}")
        ):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compress all HDF5 datasets in a directory."
    )
    parser.add_argument(
        "--dataset_dir",
        action="store",
        type=str,
        required=True,
        help="Directory containing the uncompressed datasets.",
    )

    args = DatasetCompressor(parser.parse_args())

    output_dataset_dir = args.dataset_dir + "_compressed"
    os.makedirs(output_dataset_dir, exist_ok=True)