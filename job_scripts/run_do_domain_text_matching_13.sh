#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=23:59:59
#$ -N do_domain_text_matching_13
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python do_domain_text_matching.py ../vislang-domain-exploration-data/Experiments/experiment_TextMatchingDomainBalanceParams ../vislang-domain-exploration-data/LAION-5B-Subset/ImageEmbeddingsAndMetadata 13 16

