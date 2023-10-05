#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 5
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -l h_rt=5:59:59
#$ -N trainevalplot_CC10DomainlessTextProp50DisentanglementLambda0_1_hard_zeroshot
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
#python train_clip_adapter_corrupted_cifar10.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CC10DomainlessTextProp50DisentanglementLambda0_1 ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-train-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/train/class_domain_dict.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl hard_zeroshot 0 2
#python evaluate_checkpoints_corrupted_cifar10.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CC10DomainlessTextProp50DisentanglementLambda0_1 ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/train/class_domain_dict.pkl 0 hard_zeroshot 0
#python evaluate_checkpoints_corrupted_cifar10.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CC10DomainlessTextProp50DisentanglementLambda0_1 ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/train/class_domain_dict.pkl 1 hard_zeroshot 0
#python evaluate_checkpoints_corrupted_cifar10.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CC10DomainlessTextProp50DisentanglementLambda0_1 ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/train/class_domain_dict.pkl 0 hard_zeroshot 1
#python evaluate_checkpoints_corrupted_cifar10.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CC10DomainlessTextProp50DisentanglementLambda0_1 ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/train/class_domain_dict.pkl 1 hard_zeroshot 1
#python evaluate_checkpoints_corrupted_cifar10.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CC10DomainlessTextProp50DisentanglementLambda0_1 ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/train/class_domain_dict.pkl 0 hard_zeroshot 2
#python evaluate_checkpoints_corrupted_cifar10.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CC10DomainlessTextProp50DisentanglementLambda0_1 ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/train/class_domain_dict.pkl 1 hard_zeroshot 2
python make_corrupted_cifar10_plots.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CC10DomainlessTextProp50DisentanglementLambda0_1 ../vislang-domain-exploration-data/ToyExperiments/plots/BatchSize64_plots/plot_CC10DomainlessTextProp50DisentanglementLambda0_1_hard_zeroshot.png 0 hard_zeroshot
python make_corrupted_cifar10_plots.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CC10DomainlessTextProp50DisentanglementLambda0_1 ../vislang-domain-exploration-data/ToyExperiments/plots/BatchSize64_plots/plot_CC10DomainlessTextProp50DisentanglementLambda0_1_DomainPred_hard_zeroshot.png 1 hard_zeroshot
