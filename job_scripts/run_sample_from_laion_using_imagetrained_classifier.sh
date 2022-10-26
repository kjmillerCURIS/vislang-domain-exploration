#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=2:59:59
#$ -N sample_from_laion_using_imagetrained_classifier
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python sample_from_laion.py ../vislang-domain-exploration-data/Experiments/experiment_ImageTrainedDomainBalanceParams ../vislang-domain-exploration-data/LAION-5B-Subset/ImageEmbeddingsAndMetadata

