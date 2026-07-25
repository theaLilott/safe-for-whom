#!/bin/bash
# Usage: bash jobs/submit_rq3_pipeline.sh
# Run from project root directory

cd /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy
mkdir -p logs

JOB1=$(sbatch jobs/rq3_intervene.sh | awk '{print $NF}')
echo "rq3_intervene: $JOB1"

JOB2=$(sbatch --dependency=afterok:$JOB1 jobs/rq3_evaluate.sh  | awk '{print $NF}')
echo "rq3_evaluate:     $JOB2"


echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u mkemperm"
echo "  tail -f logs/prepare_${JOB1}.out"
echo "  tail -f logs/generate_${JOB2}.out"
