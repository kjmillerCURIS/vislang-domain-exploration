#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=2:59:59
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -N compute_CLIP_paper_zeroshot_classifier
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd data/vislang-domain-exploration
python compute_CLIP_paper_zeroshot_classifier.py ../vislang-domain-exploration-data/official_CLIP_paper_zeroshot_classifier_ImageNet1k.npy


