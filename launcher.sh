#!/bin/bash -e
#SBATCH --job-name="p9_train_encoder"
#SBATCH -D .
#SBATCH --output=p9_files_out/p9_train_%j.out
#SBATCH --error=p9_files_out/p9_train_%j.err
#SBATCH --ntasks=1
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=160
#SBATCH --qos=debug

NUM_EPOCHS=5
GPUS=1
BATCH_SIZE=2*${GPUS}

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUNDIR=$HOME/politica/cmp_encoder
RESULTSDIR=$RUNDIR/files_out/

# Define application variables
exec_file=$RUNDIR/train_encoder.py

TYPE="complete" #reduced | complete

# Run job
python3 -u "$exec_file" --epochs ${NUM_EPOCHS} --print-graphs > $RESULTSDIR/${TYPE}_epochs_${NUM_EPOCHS}_gpus_${GPUS}.out

