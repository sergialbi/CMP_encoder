#!/bin/bash -e
#SBATCH --job-name="finetuning_cmp"
#SBATCH --qos=acc_debug
#SBATCH --account=bsc48
#SBATCH --output=/home/bsc/bsc048726/politics/new_cmp_encoder/results/finetuning/slurm_out/train_debug_%j.out
#SBATCH --error=/home/bsc/bsc048726/politics/new_cmp_encoder/results/finetuning/slurm_out/train_debug_%j.err
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40

NUM_EPOCHS=2
GPUS=1
BATCH_SIZE=$(expr 16 '*' ${GPUS})

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUNDIR=$HOME/politics/new_cmp_encoder
RESULTSDIR=$RUNDIR/results/finetuning/

# Define application variables
exec_file=$RUNDIR/src/finetuning/train_encoder.py
data_path="/home/bsc/bsc048726/politics/new_cmp_encoder/datasets/cmp/international"
context=1
perc="" #"a5" | "a10" | "a20"
train_file_name="CMPDb${perc}_c${context}_train.csv"
model="base" #pretrained | base

TYPE="reduced" #reduced | complete
# Run job
module load anaconda cuda/12.1 nccl/2.20.5 cudnn/9.1.0-cuda12 && \
source activate politics_new && \
python3 -u "$exec_file" --epochs ${NUM_EPOCHS} --reduced --reduced-size 1000 --print-graphs --bs ${BATCH_SIZE} --data-path ${data_path} --train-file-name ${train_file_name} > $RESULTSDIR/${TYPE}_${model}_epochs_${NUM_EPOCHS}_gpus_${GPUS}.out

