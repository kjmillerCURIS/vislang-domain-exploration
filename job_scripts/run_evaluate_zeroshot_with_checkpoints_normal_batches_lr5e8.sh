#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=23:59:59
#$ -N evaluate_zeroshot_with_checkpoints_normal_batches_lr5e8
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python evaluate_zeroshot_with_checkpoints.py ../vislang-domain-exploration-data/Experiments/experiment_NormalBatchingParamsLR5e8 ../vislang-domain-exploration-data/ILSVRC2012_val

