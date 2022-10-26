#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l h_rt=11:59:59
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -N compute_domain_log_probs_on_laion_text_0
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python compute_domain_log_probs_on_laion_text.py ../vislang-domain-exploration-data/Experiments/experiment_TextTrainedTextInferenceDomainBalanceParams ../vislang-domain-exploration-data/LAION-5B-Subset/ImageEmbeddingsAndMetadata 0 6

