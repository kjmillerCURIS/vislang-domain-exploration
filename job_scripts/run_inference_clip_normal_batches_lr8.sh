#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l h_rt=11:59:59
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -N inference_clip_normal_batches_lr8
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python inference_clip_checkpoints.py ../vislang-domain-exploration-data/Experiments/experiment_NormalBatchingParamsLR8 000-000000000,000-000000002,000-000000004,000-000000008,000-000000016,000-000000032,000-000000063,000-000000126,000-000000253,000-000000506,000-000001012,000-000001517,001-000000000,001-000000506,001-000001012,001-000001517,002-000000000,003-000000000,004-000000000,005-000000000,006-000000000,007-000000000,008-000000000,009-000000000,FINAL ../vislang-domain-exploration-data/ILSVRC2012_val

