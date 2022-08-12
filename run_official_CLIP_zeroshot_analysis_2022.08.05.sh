#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=0:59:59
#$ -N official_CLIP_zeroshot_analysis
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd data/vislang-domain-exploration
python do_official_CLIP_zeroshot_analysis.py ../vislang-domain-exploration-data/ILSVRC2012_val ../vislang-domain-exploration-data/CLIP_embeddings-ImageNet1kVal_handcrafted_augs-handcrafted_prompts/CLIP_embeddings-ImageNet1kVal_handcrafted_augs-handcrafted_prompts ../vislang-domain-exploration-data/official_CLIP_paper_zeroshot_classifier_ImageNet1k.npy ../vislang-domain-exploration-data/CLIP-ImageNet1kVal_handcrafted_augs-handcrafted_prompts-official_CLIP_zeroshot_stats_dict.pkl


