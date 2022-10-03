#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=23:59:59
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -N finetune_clip_smallbatch
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python finetune_clip_on_domain_pure_batches.py ../vislang-domain-exploration-data/Experiments/experiment_TextTrainedDomainBalanceSmallBatchParams 0

