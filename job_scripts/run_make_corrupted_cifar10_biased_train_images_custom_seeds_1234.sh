#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=2:59:59
#$ -N make_corrupted_cifar10_biased_train_images_custom_seeds_1234
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python make_corrupted_cifar10_biased_train_images_custom_seed.py ../vislang-domain-exploration-data/CIFAR10 ../vislang-domain-exploration-data/CorruptedCIFAR10 1
python make_corrupted_cifar10_biased_train_images_custom_seed.py ../vislang-domain-exploration-data/CIFAR10 ../vislang-domain-exploration-data/CorruptedCIFAR10 2
python make_corrupted_cifar10_biased_train_images_custom_seed.py ../vislang-domain-exploration-data/CIFAR10 ../vislang-domain-exploration-data/CorruptedCIFAR10 3
python make_corrupted_cifar10_biased_train_images_custom_seed.py ../vislang-domain-exploration-data/CIFAR10 ../vislang-domain-exploration-data/CorruptedCIFAR10 4

