#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=0:59:59
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -N clip_GPU_stress_test
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python clip_GPU_stress_test.py

