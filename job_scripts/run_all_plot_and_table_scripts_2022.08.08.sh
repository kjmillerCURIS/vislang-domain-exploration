#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=0:14:59
#$ -N tablify_subspace_stats_dict
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd data/vislang-domain-exploration

python make_robustness_plots.py ../vislang-domain-exploration-data/CLIP-ImageNet1kVal_handcrafted_augs-handcrafted_prompts-robustness_and_recovery_stats_dict.pkl ../vislang-domain-exploration-data/plots/CLIP-ImageNet1kVal_handcrafted_augs-handcrafted_prompts-robustness_and_recovery

python make_subspace_plots.py ../vislang-domain-exploration-data/CLIP-ImageNet1kVal_handcrafted_augs-handcrafted_prompts-subspace_stats_dict.pkl ../vislang-domain-exploration-data/plots/CLIP-ImageNet1kVal_handcrafted_augs-handcrafted_prompts-subspace

python tablify_robustness_and_recovery_stats_dict.py ../vislang-domain-exploration-data/CLIP-ImageNet1kVal_handcrafted_augs-handcrafted_prompts-robustness_and_recovery_stats_dict.pkl ../vislang-domain-exploration-data/tables/CLIP-ImageNet1kVal_handcrafted_augs-handcrafted_prompts-robustness_and_recovery_stats_table.csv

python tablify_official_CLIP_zeroshot_stats_dict.py ../vislang-domain-exploration-data/CLIP-ImageNet1kVal_handcrafted_augs-handcrafted_prompts-official_CLIP_zeroshot_stats_dict.pkl ../vislang-domain-exploration-data/tables/CLIP-ImageNet1kVal_handcrafted_augs-handcrafted_prompts-official_CLIP_zeroshot_stats_table.csv

python tablify_subspace_stats_dict.py ../vislang-domain-exploration-data/CLIP-ImageNet1kVal_handcrafted_augs-handcrafted_prompts-subspace_stats_dict.pkl ../vislang-domain-exploration-data/tables/CLIP-ImageNet1kVal_handcrafted_augs-handcrafted_prompts-subspace_stats_table.csv


