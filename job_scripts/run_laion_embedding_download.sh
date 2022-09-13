#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=11:59:59
#$ -N laion_embedding_download
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python download_laion5b_subset_embeddings.py ../vislang-domain-exploration-data/LAION-5B-Subset/ImageEmbeddingsAndMetadata

