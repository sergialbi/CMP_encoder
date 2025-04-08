#!/bin/bash -e
#SBATCH --job-name="p9_train_encoder"
#SBATCH -D .
#SBATCH --output=files_out/p9_train_%j.out
#SBATCH --error=files_out/p9_train_%j.err
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=80
#SBATCH --qos=debug

NUM_EPOCHS=1
GPUS=2
BATCH_SIZE=4*${GPUS}

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUNDIR=$HOME/politics/cmp_encoder
RESULTSDIR=$RUNDIR/files_out/

# Define application variables
exec_file=$RUNDIR/train_encoder_multiGPU.py

TYPE="reduced" #reduced | complete
#TYPE="complete"
# Run job
python3 -u "$exec_file" --epochs ${NUM_EPOCHS} --reduced --reduced-size 1000 --world-size 2 --print-graphs > $RESULTSDIR/${TYPE}_epochs_${NUM_EPOCHS}_gpus_${GPUS}.out

