import os
import sys
from script_writing_utils import write_script

#write_script(cmds, duration, num_cpus, num_gpus, job_name, script_filename)

NUM_SPLITS = 3
NUM_CPUS = 3
NUM_GPUS = 1
DURATION = '1:59:59'

def launch_disentanglement_eval(params_key):
    experiment_dir = '../vislang-domain-exploration-data/ToyExperiments/experiment_' + params_key
    assert(os.path.exists(os.path.join('..', experiment_dir)))
    script_dir = params_key
    assert(os.path.exists(script_dir))

    script_filenames = []

    #write trivial eval script
    cmds = []
    for swap_class_and_domain in [0,1]:
        cmds.append('python evaluate_checkpoints_corrupted_cifar10.py %s ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/train/class_domain_dict.pkl %d trivial -1'%(experiment_dir, swap_class_and_domain))

    write_script(cmds, DURATION, NUM_CPUS, NUM_GPUS, params_key + '-eval-trivial', os.path.join(script_dir, params_key + '-eval-trivial.sh'))
    script_filenames.append(os.path.join(script_dir, params_key + '-eval-trivial.sh'))

    #write hard_zeroshot eval scripts
    for split_index in range(NUM_SPLITS):
        cmds = []
        for swap_class_and_domain in [0,1]:
            cmds.append('python evaluate_checkpoints_corrupted_cifar10.py %s ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/train/class_domain_dict.pkl %d hard_zeroshot %d'%(experiment_dir, swap_class_and_domain, split_index))

        write_script(cmds, DURATION, NUM_CPUS, NUM_GPUS, params_key + '-eval-hard_zeroshot-split_%d'%(split_index), os.path.join(script_dir, params_key + '-eval-hard_zeroshot-split_%d.sh'%(split_index)))
        script_filenames.append(os.path.join(script_dir, params_key + '-eval-hard_zeroshot-split_%d.sh'%(split_index)))

    #write easy_zeroshot eval scripts
    for split_index in range(NUM_SPLITS):
        cmds = []
        for swap_class_and_domain in [0,1]:
            cmds.append('python evaluate_checkpoints_corrupted_cifar10.py %s ../vislang-domain-exploration-data/CorruptedCIFAR10-group_splits.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-test-images.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10-CLIP_adapter_inputs-ViTB32-text.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/test/class_domain_dict.pkl ../vislang-domain-exploration-data/CorruptedCIFAR10/train/class_domain_dict.pkl %d easy_zeroshot %d'%(experiment_dir, swap_class_and_domain, split_index))

        write_script(cmds, DURATION, NUM_CPUS, NUM_GPUS, params_key + '-eval-easy_zeroshot-split_%d'%(split_index), os.path.join(script_dir, params_key + '-eval-easy_zeroshot-split_%d.sh'%(split_index)))
        script_filenames.append(os.path.join(script_dir, params_key + '-eval-easy_zeroshot-split_%d.sh'%(split_index)))

    #fire off jobs
    for script_filename in script_filenames:
        os.system('qsub ' + script_filename)

def usage():
    print('Usage: python launch_disentanglement_eval.py <params_key>')

if __name__ == '__main__':
    launch_disentanglement_eval(*(sys.argv[1:]))
