#!/bin/bash
#SBATCH -J prepare_datasets
#SBATCH -o /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy/logs/prepare_%j.out
#SBATCH -e /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy/logs/prepare_%j.err
#SBATCH -D /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy
#SBATCH --partition=h200,h100sxm,h100nvl   # try all high-mem partitions
#SBATCH --gres=gpu:1                        # don't specify gpu type — let scheduler pick
#SBATCH -c 12
#SBATCH -N 1
#SBATCH --mem=50G
#SBATCH -t 0-16:00           # translation of full dataset — give it plenty of time
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkemperm@mpi-sws.org

source /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy/venv/bin/activate
export HF_HOME="/SWS/llms/nobackup"

mkdir -p logs

srun python3 prepare_datasets.py \
    --model Qwen/Qwen3-32B \
    --gpu 0 \
    --n_samples 600 \
    --languages spanish italian arabic japanese russian thai \
    --output_dir datasets \
    --batch_checkpoint_every 50