#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l h_rt=23:59:59
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -N inference_clip_normal_batches_english_only_lr6
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python inference_clip_checkpoints.py ../vislang-domain-exploration-data/Experiments/experiment_EnglishOnlyNormalBatchingParamsLR6 000-000000000,000-000000002,000-000000004,000-000000009,000-000000018,000-000000035,000-000000071,000-000000142,000-000000284,000-000000568,000-000001136,000-000001703,001-000000000,001-000000568,001-000001136,001-000001703,002-000000000,003-000000000,004-000000000,005-000000000,006-000000000,007-000000000,008-000000000,009-000000000,FINAL ../vislang-domain-exploration-data/ILSVRC2012_val

