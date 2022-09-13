#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l h_rt=11:59:59
#$ -N fit_linear_probes_2
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python fit_linear_probes.py ../vislang-domain-exploration-data/ILSVRC2012_train ../vislang-domain-exploration-data/CLIP_embeddings-ImageNet1kTrain_handcrafted_augs-handcrafted_prompts/CLIP_embeddings-ImageNet1kTrain_handcrafted_augs-handcrafted_prompts ../vislang-domain-exploration-data/CLIP_linearprobes-ImageNet1kTrain_handcrafted_augs 2 5

