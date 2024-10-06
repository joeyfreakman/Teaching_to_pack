#!/bin/bash

python /root/Teaching_to_pack/src/core/curr_image/curr_image_run.py \
    --task_name task1 \
    --ckpt_dir /mnt/d/kit/ALR/dataset/test149/curr_test \
    --policy_class Diffusion \
    --chunk_size 16 \
    --batch_size 1 \
    --num_epochs 100 \
    --lr 1e-4 \
    --seed 42 \
    --gpu 0 \
    --log_wandb \
    --dataset_dir /mnt/d/kit/ALR/dataset/ttp_compressed/ \