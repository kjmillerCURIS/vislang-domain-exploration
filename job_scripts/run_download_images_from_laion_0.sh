#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 4
#$ -l h_rt=23:59:59
#$ -N download_images_from_laion_0
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python download_images_from_laion.py ../vislang-domain-exploration-data/Experiments/experiment_TextTrainedDomainBalanceParams 0,1,2

