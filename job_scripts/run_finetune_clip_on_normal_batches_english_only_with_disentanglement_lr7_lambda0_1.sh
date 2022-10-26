#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l h_rt=23:59:59
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -N finetune_clip_on_normal_batches_english_only_with_disentanglement_lr7_lambda0_1
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python finetune_clip_on_normal_batches_with_disentanglement.py ../vislang-domain-exploration-data/Experiments/experiment_EnglishOnlyDisentanglementParamsLR7Lambda0_1 ../vislang-domain-exploration-data/ILSVRC2012_val 2

