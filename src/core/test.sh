#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/root/Teaching_to_pack/src/aloha/aloha_scripts: /root/interbotix_ws: /root/Teaching_to_pack"
python /root/Teaching_to_pack/src/core/test_image.py \
    --task_name task1 \
    --ckpt_dir /mnt/d/kit/ALR/dataset/test149 \
    --policy_class Diffusion \
    --chunk_size 16 \
    --batch_size 1 \
    --num_epochs 100 \
    --lr 1e-4 \
    --seed 42 \
    --gpu 0 \
    --log_wandb \
    --test \