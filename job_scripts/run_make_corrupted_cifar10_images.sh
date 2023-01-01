#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=2:59:59
#$ -N make_corrupted_cifar10_images
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python make_corrupted_cifar10_images.py ../vislang-domain-exploration-data/CIFAR10 ../vislang-domain-exploration-data/CorruptedCIFAR10

