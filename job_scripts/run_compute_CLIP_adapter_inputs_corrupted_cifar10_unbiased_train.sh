#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -l h_rt=5:59:59
#$ -N compute_CLIP_adapter_inputs_corrupted_cifar10_unbiased_train
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python compute_CLIP_adapter_inputs_corrupted_cifar10_unbiased_train.py ../vislang-domain-exploration-data/CorruptedCIFAR10 ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32

