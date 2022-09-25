#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=0:59:59
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -N train_domain_classifier_linear
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python train_domain_classifier.py TextTrainedDomainBalanceLinearClassifierParams ../vislang-domain-exploration-data/CLIP_embeddings-ViTL14-ImageNet1kTrain_handcrafted_augs-handcrafted_prompts/CLIP_embeddings-ViTL14-ImageNet1kTrain_handcrafted_augs-handcrafted_prompts ../vislang-domain-exploration-data/Experiments/experiment_TextTrainedDomainBalanceLinearClassifierParams

