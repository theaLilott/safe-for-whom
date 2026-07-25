#!/bin/bash
#SBATCH -J rq3_intervene
#SBATCH -o /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy/logs/intervene_%j.out
#SBATCH -e /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy/logs/intervene_%j.err
#SBATCH -D /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy
#SBATCH --partition=h200,h100sxm,h100nvl
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -N 1
#SBATCH --mem=50G
#SBATCH -t 0-16:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkemperm@mpi-sws.org

source /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy/venv/bin/activate
export HF_HOME="/SWS/llms/nobackup"

pip install jaxtyping

mkdir -p logs

srun python3 rq3_causal.py \
    --model Qwen/Qwen3-14B \
    --gpu 0 \
    --languages english german indonesian \
    --dataset_dir datasets \
    --output_dir results \
    --scale -1.0 \
    --intervention_layer 24 \
    --checkpoint_every 25