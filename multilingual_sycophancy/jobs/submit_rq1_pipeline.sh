#!/bin/bash
# Usage: bash jobs/submit_pipeline.sh
# Run from project root directory

cd /NS/MAS-llms01/nobackup/mkemperm/multilingual-sycophancy
mkdir -p logs

JOB1=$(sbatch jobs/prepare_datasets.sh | awk '{print $NF}')
echo "prepare_datasets: $JOB1"

JOB2=$(sbatch --dependency=afterok:$JOB1 jobs/rq1_generate.sh  | awk '{print $NF}')
echo "rq1_generate:     $JOB2"

JOB3=$(sbatch --dependency=afterok:$JOB2 jobs/rq1_evaluate.sh | awk '{print $NF}')
echo "rq1_evaluate:     $JOB3"

#JOB4=$(sbatch  jobs/rq2_collect_activations.sh | awk '{print $NF}')
#echo "rq2_collect_activations:  $JOB4"

echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u mkemperm"
echo "  tail -f logs/prepare_${JOB1}.out"
echo "  tail -f logs/generate_${JOB2}.out"
echo "  tail -f logs/evaluate_${JOB3}.out"
#echo "  tail -f logs/activations_${JOB4}.out"