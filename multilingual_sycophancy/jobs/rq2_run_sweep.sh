#!/bin/bash
#SBATCH -J rq2_probe_sweep
#SBATCH -o /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy/logs/probe_sweep_%j.out
#SBATCH -e /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy/logs/probe_sweep_%j.err
#SBATCH -D /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy
#SBATCH --partition=spyder
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH -t 0-01:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkemperm@mpi-sws.org

source /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy/venv/bin/activate

mkdir -p logs results/probe_sweep

# Sweep all (act_set, language) combinations
for ACT_SET in avg avg_final; do
    for LANG in english german indonesian spanish italian arabic russian thai; do
        echo "=== Sweeping act_set=$ACT_SET lang=$LANG ==="
        python3 rq2_probe_sweep.py --act_set "$ACT_SET" --lang "$LANG"
    done
done

echo "=== All sweeps done, aggregating ==="
python3 rq2_aggregate_sweep.py

echo "Finished at $(date)"
