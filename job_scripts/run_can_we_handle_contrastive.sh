#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=0:59:59
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -N can_we_handle_contrastive
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python can_we_handle_contrastive.py

