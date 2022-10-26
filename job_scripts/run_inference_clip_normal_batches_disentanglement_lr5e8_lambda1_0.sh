#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l h_rt=23:59:59
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -N inference_clip_normal_batches_disentanglement_lr5e8_lambda1_0
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python inference_clip_checkpoints.py ../vislang-domain-exploration-data/Experiments/experiment_DisentanglementParamsLR5e8Lambda1_0 ALL ../vislang-domain-exploration-data/ILSVRC2012_val

