export PYTHONPATH="${PYTHONPATH}:/root/Teaching_to_pack/src/aloha/aloha_scripts: /root/interbotix_ws: /root/Teaching_to_pack"
python /root/Teaching_to_pack/src/core/speed_image/speed_run.py \
    --task_name task1 \
    --ckpt_dir /root/Teaching_to_pack/environment/dataset/ll_ckpt/task1 \
    --policy_class Diffusion \
    --chunk_size 16 \
    --batch_size 1 \
    --num_epochs 100 \
    --lr 1e-4 \
    --seed 42 \
    --gpu 0 \
    --log_wandb \
    --dataset_dir /mnt/d/kit/ALR/dataset/ttp_compressed \
