#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=0:14:59
#$ -N tablify_subspace_stats_dict
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd data/vislang-domain-exploration
python tablify_subspace_stats_dict.py ../vislang-domain-exploration-data/CLIP-ImageNet1kVal_handcrafted_augs-handcrafted_prompts-subspace_stats_dict.pkl ../vislang-domain-exploration-data/CLIP-ImageNet1kVal_handcrafted_augs-handcrafted_prompts-subspace_stats_table.csv


