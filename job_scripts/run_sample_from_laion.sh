#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=5:59:59
#$ -N sample_from_laion
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python sample_from_laion.py ../vislang-domain-exploration-data/Experiments/experiment_TextTrainedDomainBalanceParams ../vislang-domain-exploration-data/LAION-5B-Subset/ImageEmbeddingsAndMetadata

