#!/bin/bash
#SBATCH -J rq1_generate
#SBATCH -o /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy/logs/generate_%j.out
#SBATCH -e /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy/logs/generate_%j.err
#SBATCH -D /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy
#SBATCH --partition=h200,h100sxm,h100nvl   # try all high-mem partitions
#SBATCH --gres=gpu:1                        # don't specify gpu type — let scheduler pick
#SBATCH -c 12
#SBATCH -N 1
#SBATCH --mem=50G
#SBATCH -t 0-16:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkemperm@mpi-sws.org

source /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy/venv/bin/activate
export HF_HOME="/SWS/llms/nobackup"

mkdir -p logs

srun python3 rq1_generate.py \
    --model Qwen/Qwen3-14B \
    --gpu 0 \
    --languages spanish italian arabic japanese russian thai \
    --dataset_dir datasets \
    --output_dir results \
    --split test \
    --checkpoint_every 25