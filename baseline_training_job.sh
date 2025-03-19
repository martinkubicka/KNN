#!/bin/bash
#PBS -N knn_job
#PBS -l select=1:ncpus=1:mem=64gb:ngpus=1:gpu_mem=16gb:scratch_local=32gb
#PBS -l walltime=100:00:00
#PBS -q gpu_long

DATADIR=/storage/brno2/home/xkubic45/KNN
cd $DATADIR || { echo "Failed to change directory to $DATADIR" >&2; exit 2; }
chmod +x setup.sh test_baseline.sh train_baseline.sh || { echo "Failed to change permissions" >&2; exit 3; }
./setup.sh || { echo "setup.sh failed" >&2; exit 4; }
python -u ./src/architectures/baseline/train.py > output.txt 2>&1
