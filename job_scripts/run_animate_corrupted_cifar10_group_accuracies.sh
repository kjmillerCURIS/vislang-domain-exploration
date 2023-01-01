#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=2:59:59
#$ -N animate_corrupted_cifar10_group_accuracies
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
python animate_corrupted_cifar10_group_accuracies.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CorruptedCIFAR10BaselineParamsLR5 ../vislang-domain-exploration-data/CorruptedCIFAR10
python animate_corrupted_cifar10_group_accuracies.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CorruptedCIFAR10BaselineParamsLR5_UnbiasedTrain ../vislang-domain-exploration-data/CorruptedCIFAR10
python animate_corrupted_cifar10_group_accuracies.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CorruptedCIFAR10BaselineParamsLR5_TrainBiasSamplingSeed1 ../vislang-domain-exploration-data/CorruptedCIFAR10
python animate_corrupted_cifar10_group_accuracies.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CorruptedCIFAR10BaselineParamsLR5_TrainBiasSamplingSeed2 ../vislang-domain-exploration-data/CorruptedCIFAR10
python animate_corrupted_cifar10_group_accuracies.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CorruptedCIFAR10BaselineParamsLR5_TrainBiasSamplingSeed3 ../vislang-domain-exploration-data/CorruptedCIFAR10
python animate_corrupted_cifar10_group_accuracies.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CorruptedCIFAR10BaselineParamsLR5_TrainBiasSamplingSeed4 ../vislang-domain-exploration-data/CorruptedCIFAR10

