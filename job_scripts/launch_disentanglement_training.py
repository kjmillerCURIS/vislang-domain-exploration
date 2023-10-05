import os
import sys
from script_writing_utils import write_script

#write_script(cmds, duration, num_cpus, num_gpus, job_name, script_filename)

NUM_SPLITS = 3
NUM_CPUS = 3
NUM_GPUS = 1
DURATION = '7:59:59'

def launch_disentanglement_training(params_key):
    experiment_dir = '../vislang-domain-exploration-data/ToyExperiments/experiment_' + params_key
    assert(os.path.exists(os.path.join('..', experiment_dir)))
    script_dir = params_key
    os.makedirs(script_dir, exist_ok=True)

    script_filenames = []

    #write trivial training script
    write_script(['python train_clip_adapter_corrupted_cifar10.py %s ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-train-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/train/class_domain_dict.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl trivial -1 1'%(experiment_dir)], DURATION, NUM_CPUS, NUM_GPUS, params_key + '-train-trivial', os.path.join(script_dir, params_key + '-train-trivial.sh'))
    script_filenames.append(os.path.join(script_dir, params_key + '-train-trivial.sh'))

    #write hard_zeroshot training scripts
    for split_index in range(NUM_SPLITS):
        write_script(['python train_clip_adapter_corrupted_cifar10.py %s ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-train-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/train/class_domain_dict.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl hard_zeroshot %d 1'%(experiment_dir, split_index)], DURATION, NUM_CPUS, NUM_GPUS, params_key + '-train-hard_zeroshot-split_%d'%(split_index), os.path.join(script_dir, params_key + '-train-hard_zeroshot-split_%d.sh'%(split_index)))
        script_filenames.append(os.path.join(script_dir, params_key + '-train-hard_zeroshot-split_%d.sh'%(split_index)))

    #write easy_zeroshot training scripts
    for split_index in range(NUM_SPLITS):
        write_script(['python train_clip_adapter_corrupted_cifar10.py %s ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-train-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/train/class_domain_dict.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl easy_zeroshot %d 1'%(experiment_dir, split_index)], DURATION, NUM_CPUS, NUM_GPUS, params_key + '-train-easy_zeroshot-split_%d'%(split_index), os.path.join(script_dir, params_key + '-train-easy_zeroshot-split_%d.sh'%(split_index)))
        script_filenames.append(os.path.join(script_dir, params_key + '-train-easy_zeroshot-split_%d.sh'%(split_index)))

    #fire off jobs
    for script_filename in script_filenames:
        os.system('qsub ' + script_filename)

def usage():
    print('Usage: python launch_disentanglement_training.py <params_key>')

if __name__ == '__main__':
    launch_disentanglement_training(*(sys.argv[1:]))
