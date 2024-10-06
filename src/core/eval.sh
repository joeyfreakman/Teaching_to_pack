
python run.py \
    --task_name teachingtopack \
    --ckpt_dir /root/Teaching_to_pack/environment/dataset/ll_ckpt/task1 \
    --policy_class Diffusion \
    --chunk_size 16 \
    --batch_size 16 \
    --max_skill_len 200 \
    --num_epochs 1000 \
    --lr 1e-4 \
    --seed 42 \
    --temporal_agg True \
    --log_wandb \
    --multi_gpu \
    --eval \