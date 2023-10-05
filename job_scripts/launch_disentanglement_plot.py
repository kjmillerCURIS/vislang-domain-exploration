import os
import sys
from script_writing_utils import write_script
sys.path.append('..')
from make_corrupted_cifar10_plots import get_result_base_dirs

#write_script(cmds, duration, num_cpus, num_gpus, job_name, script_filename)
#get_result_base_dirs(experiment_dir, split_type, swap_class_and_domain=False)

NUM_CPUS = 1
NUM_GPUS = 0
DURATION = '1:59:59'
EXPECTED_NUM_SPLITS = 3

def check_results(experiment_dir, split_type, expected_num_splits, swap_class_and_domain=False):
    our_experiment_dir = os.path.join('..', experiment_dir)
    result_base_dirs = get_result_base_dirs(our_experiment_dir, split_type, swap_class_and_domain=swap_class_and_domain)
    expected_result_base_dirs = []
    suffix = {True : '-predict_domain', False : ''}[swap_class_and_domain]
    if split_type == 'trivial':
        expected_result_base_dirs.append('results%s'%(suffix))
    else:
        assert(split_type in ['easy_zeroshot', 'hard_zeroshot'])
        for i in range(expected_num_splits):
            expected_result_base_dirs.append('results-%s-%d%s'%(split_type, i, suffix))

    assert(sorted(result_base_dirs) == sorted(expected_result_base_dirs))
    for result_base_dir in result_base_dirs:
        final_filename = os.path.join(our_experiment_dir, result_base_dir, 'result%s-FINAL.pkl'%(suffix))
        assert(os.path.exists(final_filename))

def launch_disentanglement_plot(params_key):
    experiment_dir = '../vislang-domain-exploration-data/ToyExperiments/experiment_' + params_key
    assert(os.path.exists(os.path.join('..', experiment_dir)))
    script_dir = params_key
    assert(os.path.exists(script_dir))

    script_filenames = []

    #just one script to do everything
    script_filename = os.path.join(script_dir, params_key + '-plot.sh')
    job_name = params_key + '-plot'
    cmds = []

    #trivial commands
    for swap_class_and_domain in [0,1]:
        check_results(experiment_dir, 'trivial', EXPECTED_NUM_SPLITS, swap_class_and_domain=swap_class_and_domain)
        suffix = {True : '_DomainPred', False : ''}[swap_class_and_domain]
        cmds.append('python make_corrupted_cifar10_plots.py %s ../vislang-domain-exploration-data/ToyExperiments/plots/BatchSize64_plots/plot_%s%s.png %d trivial'%(experiment_dir, params_key, suffix, swap_class_and_domain))

    #hard_zeroshot commands
    for swap_class_and_domain in [0,1]:
        check_results(experiment_dir, 'hard_zeroshot', EXPECTED_NUM_SPLITS, swap_class_and_domain=swap_class_and_domain)
        suffix = {True : '_DomainPred', False : ''}[swap_class_and_domain]
        cmds.append('python make_corrupted_cifar10_plots.py %s ../vislang-domain-exploration-data/ToyExperiments/plots/BatchSize64_plots/plot_%s%s_hard_zeroshot.png %d hard_zeroshot'%(experiment_dir, params_key, suffix, swap_class_and_domain))

    #easy_zeroshot commands
    for swap_class_and_domain in [0,1]:
        check_results(experiment_dir, 'easy_zeroshot', EXPECTED_NUM_SPLITS, swap_class_and_domain=swap_class_and_domain)
        suffix = {True : '_DomainPred', False : ''}[swap_class_and_domain]
        cmds.append('python make_corrupted_cifar10_plots.py %s ../vislang-domain-exploration-data/ToyExperiments/plots/BatchSize64_plots/plot_%s%s_easy_zeroshot.png %d easy_zeroshot'%(experiment_dir, params_key, suffix, swap_class_and_domain))

    write_script(cmds, DURATION, NUM_CPUS, NUM_GPUS, job_name, script_filename)

    script_filenames.append(script_filename)

    #fire off job(s)
    for script_filename in script_filenames:
        os.system('qsub ' + script_filename)

def usage():
    print('Usage: python launch_disentanglement_plot.py <params_key>')

if __name__ == '__main__':
    launch_disentanglement_plot(*(sys.argv[1:]))
