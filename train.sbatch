#!/bin/bash
#SBATCH --job-name=sysbinder
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/pfs/work9/workspace/scratch/ul_dqm81-sysbinder_data/logs/slurm_%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/ul_dqm81-sysbinder_data/logs/slurm_%j.err

WORKSPACE=/pfs/work9/workspace/scratch/ul_dqm81-sysbinder_data
LOG_DIR=$WORKSPACE/logs
CHECKPOINT=$WORKSPACE/checkpoint.pt.tar

mkdir -p $LOG_DIR

# Copy data to fast local scratch
echo "Copying data to TMPDIR..."
mkdir -p $TMPDIR/clevr-easy
cp -r $WORKSPACE/clevr-easy/train/ $TMPDIR/clevr-easy/train/
echo "Done. $(find $TMPDIR/clevr-easy/train -name '*.png' | wc -l) images."

module load devel/cuda/12.8

export PATH="$HOME/.local/bin:$PATH"

cd ~/sysbinder

uv run python train.py \
    --data_path "$TMPDIR/clevr-easy/train/*.png" \
    --log_path $LOG_DIR \
    --checkpoint_path $CHECKPOINT \
    --batch_size 160 \
    --num_workers 48
