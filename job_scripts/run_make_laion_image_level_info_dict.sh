#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=2:59:59
#$ -N make_laion_image_level_info_dict
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python make_laion_image_level_info_dict.py ../vislang-domain-exploration-data/LAION-5B-Subset/ImageEmbeddingsAndMetadata

