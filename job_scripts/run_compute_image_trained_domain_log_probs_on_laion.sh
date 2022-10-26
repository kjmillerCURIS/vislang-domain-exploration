#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 4
#$ -l h_rt=23:59:59
#$ -l gpus=4
#$ -l gpu_c=5.0
#$ -N compute_image_trained_domain_log_probs_on_laion
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python compute_domain_log_probs_on_laion.py ../vislang-domain-exploration-data/Experiments/experiment_ImageTrainedDomainBalanceParams ../vislang-domain-exploration-data/LAION-5B-Subset/ImageEmbeddingsAndMetadata

