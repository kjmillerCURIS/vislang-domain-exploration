#!/bin/bash -l

#$ -P ivc-ml
#$ -l cpu_arch=skylake
#$ -l gpus=1
#$ -l gpu_c=5.0
#$ -pe omp 3
#$ -l h_rt=2:59:59
#$ -N do_corrupted_cifar10_subspace_analysis
#$ -j y
#$ -m ea

module load miniconda
conda activate vislang-domain-exploration
cd ~/data/vislang-domain-exploration
#python do_corrupted_cifar10_subspace_analysis.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CorruptedCIFAR10BaselineParamsLR5BatchSize64DomainlessTextProp50 ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl
python do_corrupted_cifar10_subspace_analysis.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CC10DomainlessTextProp50DisentanglementOrthoLambda100_0 ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl
python do_corrupted_cifar10_subspace_analysis.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CC10DomainlessTextProp50DisentanglementLambda100_0 ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl
#python do_corrupted_cifar10_subspace_analysis.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CC10DomainlessTextProp50DisentanglementOrthoLambda10_0 ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl
#python do_corrupted_cifar10_subspace_analysis.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CC10DomainlessTextProp50DisentanglementLambda10_0 ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl
#python do_corrupted_cifar10_subspace_analysis.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CC10DomainlessTextProp50DisentanglementOrthoLambda1_0 ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl
#python do_corrupted_cifar10_subspace_analysis.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CC10DomainlessTextProp50DisentanglementLambda1_0 ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl
#python do_corrupted_cifar10_subspace_analysis.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CC10DomainlessTextProp50DisentanglementOrthoLambda0_1 ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl
#python do_corrupted_cifar10_subspace_analysis.py ../vislang-domain-exploration-data/ToyExperiments/experiment_CC10DomainlessTextProp50DisentanglementLambda0_1 ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl

