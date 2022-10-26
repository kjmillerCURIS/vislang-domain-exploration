#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l h_rt=11:59:59
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -N finetune_clip_on_normal_batches_lr7
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python finetune_clip_on_normal_batches.py ../vislang-domain-exploration-data/Experiments/experiment_NormalBatchingParamsLR7 2

