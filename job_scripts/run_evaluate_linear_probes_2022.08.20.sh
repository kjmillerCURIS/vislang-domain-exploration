#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=0:59:59
#$ -N evaluate_linear_probes
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python evaluate_linear_probes.py ../vislang-domain-exploration-data/ILSVRC2012_val ../vislang-domain-exploration-data/CLIP_embeddings-ImageNet1kVal_handcrafted_augs-handcrafted_prompts/CLIP_embeddings-ImageNet1kVal_handcrafted_augs-handcrafted_prompts ../vislang-domain-exploration-data/CLIP_linearprobes-ImageNet1kTrain_handcrafted_augs ../vislang-domain-exploration-data/CLIP_linearprobe-ImageNet1kTrainVal_stats_dict.pkl

