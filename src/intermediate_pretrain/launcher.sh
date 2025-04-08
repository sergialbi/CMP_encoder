#!/bin/bash -e
#SBATCH --job-name="intermediate_pretrain"
#SBATCH --qos=acc_bsccs
#SBATCH --account=bsc48
#SBATCH --output=/home/bsc/bsc048726/politics/cmp_encoder/results/intermediate_pretrain/slurm_out/intermediate_pretrain_%j.out
#SBATCH --error=/home/bsc/bsc048726/politics/cmp_encoder/results/intermediate_pretrain/slurm_out/intermediate_pretrain_%j.err
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40

NUM_EPOCHS=15
GPUS=1
BATCH_SIZE=$(expr 16 '*' ${GPUS})
LR=1e-5
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUNDIR=$HOME/politics/cmp_encoder
RESULTSDIR=$RUNDIR/results/intermediate_pretrain/

# Define application variables
exec_file=$RUNDIR/src/intermediate_pretrain/intermediate_pretrain.py
data_path_file="/home/bsc/bsc048726/politics/cmp_encoder/datasets/congress/congress.xlsx"


module load anaconda cuda/11.8 nccl cudnn/9.0.0-cuda11 && \
source activate politics && \
# Run job
python3 -u "$exec_file" --epochs ${NUM_EPOCHS} --mn5 --lr ${LR} --bs ${BATCH_SIZE} --data-path ${data_path_file} > $RESULTSDIR/intermediate_pretrain_epochs_${NUM_EPOCHS}.out

