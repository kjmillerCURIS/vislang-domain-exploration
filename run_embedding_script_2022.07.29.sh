#!/bin/bash -l

#$ -P ivc-ml
#$ -l h_rt=24:00:00
#$ -l gpus=1
#$ -N compute_CLIP_embeddings
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd data/vislang-domain-exploration
python compute_CLIP_embeddings.py ../vislang-domain-exploration-data/ILSVRC2012_val ../vislang-domain-exploration-data/CLIP_embeddings-ImageNet1kVal_handcrafted_augs-handcrafted_prompts/CLIP_embeddings-ImageNet1kVal_handcrafted_augs-handcrafted_prompts


