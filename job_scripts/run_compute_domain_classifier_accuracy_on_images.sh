#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=5:59:59
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -N compute_domain_classifier_accuracy_on_images
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python compute_domain_classifier_accuracy_on_images.py ../vislang-domain-exploration-data/Experiments/experiment_TextTrainedDomainBalanceParams ../vislang-domain-exploration-data/CLIP_embeddings-ViTL14-ImageNet1kVal_handcrafted_augs-handcrafted_prompts/CLIP_embeddings-ViTL14-ImageNet1kVal_handcrafted_augs-handcrafted_prompts ../vislang-domain-exploration-data/ILSVRC2012_val
python compute_domain_classifier_accuracy_on_images.py ../vislang-domain-exploration-data/Experiments/experiment_TextTrainedDomainBalanceLinearClassifierParams ../vislang-domain-exploration-data/CLIP_embeddings-ViTL14-ImageNet1kVal_handcrafted_augs-handcrafted_prompts/CLIP_embeddings-ViTL14-ImageNet1kVal_handcrafted_augs-handcrafted_prompts ../vislang-domain-exploration-data/ILSVRC2012_val
python compute_domain_classifier_accuracy_on_images.py ../vislang-domain-exploration-data/Experiments/experiment_TextTrainedDomainBalance100EpochsParams ../vislang-domain-exploration-data/CLIP_embeddings-ViTL14-ImageNet1kVal_handcrafted_augs-handcrafted_prompts/CLIP_embeddings-ViTL14-ImageNet1kVal_handcrafted_augs-handcrafted_prompts ../vislang-domain-exploration-data/ILSVRC2012_val
python compute_domain_classifier_accuracy_on_images.py ../vislang-domain-exploration-data/Experiments/experiment_TextTrainedDomainBalanceLinearClassifier100EpochsParams ../vislang-domain-exploration-data/CLIP_embeddings-ViTL14-ImageNet1kVal_handcrafted_augs-handcrafted_prompts/CLIP_embeddings-ViTL14-ImageNet1kVal_handcrafted_augs-handcrafted_prompts ../vislang-domain-exploration-data/ILSVRC2012_val

