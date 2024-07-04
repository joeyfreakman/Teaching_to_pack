#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/root/Teaching_to_pack/src/aloha/aloha_scripts: /root/interbotix_ws: /root/Teaching_to_pack" #add or modify your own python path
python /root/Teaching_to_pack/src/core/run.py \
    --task_name teachingtopack \
    --ckpt_dir /root/Teaching_to_pack/environment/dataset/ll_ckpt/task1 \
    --policy_class Diffusion \
    --chunk_size 16 \
    --batch_size 16 \
    --max_skill_len 200 \
    --num_epochs 1000 \
    --lr 1e-4 \
    --seed 42 \
    --gpu 0 \
    --log_wandb \
