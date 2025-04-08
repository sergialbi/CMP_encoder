#!/bin/bash -e
#SBATCH --job-name="predict_cmp"
#SBATCH --qos=acc_debug
#SBATCH --account=bsc48
#SBATCH --output=/home/bsc/bsc048726/politics/cmp_encoder/results/finetuning/slurm_out/predict_%j.out
#SBATCH --error=/home/bsc/bsc048726/politics/cmp_encoder/results/finetuning/slurm_out/predict_%j.err
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40

GPUS=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUNDIR=$HOME/politics/cmp_encoder
RESULTSDIR=$RUNDIR/results/finetuning/

# Define application variables
exec_file=$RUNDIR/src/finetuning/predict.py

# Run job
module load anaconda cuda/11.8 nccl cudnn/9.0.0-cuda11 && \
source activate politics && \
python3 -u "$exec_file" --mn5 > $RESULTSDIR/predictions.out

