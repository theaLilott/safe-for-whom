#!/bin/bash
#SBATCH -J rq1_evaluate
#SBATCH -o /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy/logs/evaluate_%j.out
#SBATCH -e /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy/logs/evaluate_%j.err
#SBATCH -D /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy
#SBATCH --partition=h200,h100sxm,h100nvl   # try all high-mem partitions
#SBATCH --gres=gpu:1                        # don't specify gpu type — let scheduler pick
#SBATCH -c 12
#SBATCH -N 1
#SBATCH --mem=60G
#SBATCH -t 0-08:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkemperm@mpi-sws.org

source /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy/venv/bin/activate
export HF_HOME="/SWS/llms/nobackup"

mkdir -p logs

srun python3 rq1_evaluate.py \
    --judge_model Qwen/Qwen3-32B \
    --gpu_judge 0 \
    --generations_file results/rq1_generations_qwen3_14b_new_langs.csv \
    --output_dir results \
    --checkpoint_every 25