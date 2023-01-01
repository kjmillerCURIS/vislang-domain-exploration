#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -l h_rt=0:29:59
#$ -N evalplot_corrupted_cifar10_UnbiasedTrain_domainpred
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration

python evaluate_checkpoints_corrupted_cifar10.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CorruptedCIFAR10BaselineParamsLR5_UnbiasedTrain ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/train/class_domain_dict.pkl 1
python make_corrupted_cifar10_plots.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CorruptedCIFAR10BaselineParamsLR5_UnbiasedTrain ../vislang-domain-exploration-data/ToyExperiments/plots/initial_experiment_plots/plot_CorruptedCIFAR10BaselineParamsLR5_UnbiasedTrain_DomainPred.png 1

