#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l h_rt=23:59:59
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -N compute_laion_text_embeddings_2
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python compute_laion_text_embeddings.py ../vislang-domain-exploration-data/LAION-5B-Subset/ImageEmbeddingsAndMetadata 2 6

