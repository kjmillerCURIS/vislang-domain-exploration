#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=0:59:59
#$ -N subspace_analysis
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd data/vislang-domain-exploration
python do_subspace_analysis.py ../vislang-domain-exploration-data/ILSVRC2012_val ../vislang-domain-exploration-data/CLIP_embeddings-ImageNet1kVal_handcrafted_augs-handcrafted_prompts/CLIP_embeddings-ImageNet1kVal_handcrafted_augs-handcrafted_prompts ../vislang-domain-exploration-data/CLIP-ImageNet1kVal_handcrafted_augs-handcrafted_prompts-subspace_stats_dict.pkl


