#!/bin/bash
#PBS -N knn_job
#PBS -l select=1:ncpus=1:mem=164gb:ngpus=1:gpu_mem=15gb:scratch_local=64gb
#PBS -l walltime=150:00:00
#PBS -q gpu_long

DATADIR=/storage/brno2/home/xkubic45/KNN/9
cd $DATADIR || { echo "Failed to change directory to $DATADIR" >&2; exit 2; }
chmod +x src/architectures/hybrid/setup/setup.sh src/architectures/hybrid/setup/test_baseline.sh src/architectures/hybrid/setup/train_baseline.sh || { echo "Failed to change permissions" >&2; exit 3; }
./src/architectures/hybrid/setup/setup.sh || { echo "setup.sh failed" >&2; exit 4; }
python -u src/architectures/hybrid/train.py > output.txt 2>&1
