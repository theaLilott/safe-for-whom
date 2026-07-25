#!/bin/bash
#SBATCH -J rq2_collect_activations
#SBATCH -o /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy/logs/activations_%j.out
#SBATCH -e /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy/logs/activations_%j.err
#SBATCH -D /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy
#SBATCH --partition=h200,h100sxm,h100nvl   # try all high-mem partitions
#SBATCH --gres=gpu:1                        # don't specify gpu type — let scheduler pick
#SBATCH -c 12
#SBATCH -N 1
#SBATCH --mem=60G
#SBATCH -t 0-04:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkemperm@mpi-sws.org

source /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy/venv/bin/activate
export HF_HOME="/SWS/llms/nobackup"

mkdir -p logs activations

srun python3 rq2_collect_activations.py \
    --model Qwen/Qwen3-14B \
    --gpu 0 \
    --languages english german indonesian spanish italian arabic russian thai \
    --dataset_dir datasets 