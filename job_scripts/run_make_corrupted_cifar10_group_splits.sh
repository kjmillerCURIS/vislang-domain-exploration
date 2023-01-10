#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=0:29:59
#$ -N make_corrupted_cifar10_group_splits
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python make_corrupted_cifar10_group_splits.py ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl

