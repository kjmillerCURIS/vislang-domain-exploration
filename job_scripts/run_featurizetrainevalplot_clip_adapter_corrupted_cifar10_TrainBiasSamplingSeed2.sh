#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -l h_rt=1:59:59
#$ -N featurizetrainevalplot_clip_adapter_corrupted_cifar10_TrainBiasSamplingSeed2
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python compute_CLIP_adapter_inputs_corrupted_cifar10_customized_train.py ../vislang-domain-exploration-data/CorruptedCIFAR10 train_BiasSamplingSeed2 ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32
python train_clip_adapter_corrupted_cifar10.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CorruptedCIFAR10BaselineParamsLR5_TrainBiasSamplingSeed2 ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-train_BiasSamplingSeed2-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/train_BiasSamplingSeed2/class_domain_dict.pkl 1
python evaluate_checkpoints_corrupted_cifar10.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CorruptedCIFAR10BaselineParamsLR5_TrainBiasSamplingSeed2 ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/train_BiasSamplingSeed2/class_domain_dict.pkl
python make_corrupted_cifar10_plots.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CorruptedCIFAR10BaselineParamsLR5_TrainBiasSamplingSeed2 ../vislang-domain-exploration-data/ToyExperiments/plots/trainimagebiassamplingrep_separateplots/plot_CorruptedCIFAR10BaselineParamsLR5_TrainBiasSamplingSeed2.png

