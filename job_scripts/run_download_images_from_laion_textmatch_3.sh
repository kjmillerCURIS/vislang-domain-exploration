#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 4
#$ -l h_rt=1:59:59
#$ -N download_images_from_laion_textmatch_3
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python download_images_from_laion.py ../vislang-domain-exploration-data/Experiments/experiment_TextMatchingDomainBalanceParams 9,10,11

