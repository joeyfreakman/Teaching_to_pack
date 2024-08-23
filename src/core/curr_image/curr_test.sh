#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=50gb
#SBATCH --cpus-per-task=60
#SBATCH --partition=dev_gpu_4_a100
#SBATCH --job-name=test_task
#SBATCH --gres=gpu:1
#SBATCH --mail-user=joeywang.of@gmail.com
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/kit/stud/utfvv/Teaching_to_pack/src/outputs/slurm-%j.out
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Running on ${SLURM_CPUS_PER_TASK} CPUs with ${OMP_NUM_THREADS} OpenMP threads"
tar -C $TMPDIR -xvzf /home/kit/stud/utfvv/compressed_hdf5.tar.gz
python /home/kit/stud/utfvv/Teaching_to_pack/src/core/curr_image/curr_test.py \
    --task_name task1 \
    --ckpt_dir /home/kit/stud/utfvv/Teaching_to_pack/environment/dataset/ll_ckpt/task_curr_image/model_ckpt \
    --policy_class Diffusion \
    --chunk_size 16 \
    --batch_size 1 \
    --num_epochs 100 \
    --lr 1e-4 \
    --seed 42 \
    --gpu 0 \
    --log_wandb \
    --test \
    --dataset_dir $TMPDIR/home/kit/stud/utfvv/Teaching_to_pack/environment/data/