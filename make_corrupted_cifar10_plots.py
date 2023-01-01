import os
import sys
import glob
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from experiment_params.param_utils import get_params_key
from experiment_params.corrupted_cifar10_params import grab_params
from command_history_utils import write_to_history

TRAINING_SET_SIZE = 50000

def get_epoch_length(experiment_dir):
    p = grab_params(get_params_key(experiment_dir))
    return TRAINING_SET_SIZE // p.clip_batch_size

#returns xs, acc_dicts
def get_acc_dicts(result_filenames, epoch_length):
    pairs = []
    for result_filename in tqdm(result_filenames):
        with open(result_filename, 'rb') as f:
            result = pickle.load(f)

        x = result['epoch'] + result['step_within_epoch'] / epoch_length
        pairs.append((x, result['acc_dict']))

    pairs = sorted(pairs, key = lambda p: p[0])
    xs = [p[0] for p in pairs]
    acc_dicts = [p[1] for p in pairs]
    return xs, acc_dicts

#experiment_dirs can be a comma-separated list, or just one dir, or it could be an actual list
def make_corrupted_cifar10_plots(experiment_dirs, plot_filename, swap_class_and_domain=False):
    write_to_history(os.path.dirname(plot_filename))
    swap_class_and_domain = int(swap_class_and_domain)
    key_ensemble = 'avg_domains'
    key_oracle = 'own_domain'
    if swap_class_and_domain:
        key_ensemble = 'avg_classes'
        key_oracle = 'own_class'

    if not isinstance(experiment_dirs, list):
        experiment_dirs = experiment_dirs.split(',')

    plt.clf()
    plt.figure(figsize=[14.4, 4.8])
    should_add_to_legend = True
    for experiment_dir in experiment_dirs:
        epoch_length = get_epoch_length(experiment_dir)
        result_base_dir = 'results'
        if swap_class_and_domain:
            result_base_dir = 'results-predict_domain'

        result_filenames = sorted(glob.glob(os.path.join(experiment_dir, result_base_dir, '*.pkl')))
        xs, acc_dicts = get_acc_dicts(result_filenames, epoch_length)
        for bias_type in ['biased', 'unbiased']:
            for prompt_type in [key_ensemble, key_oracle]:
                color = {'biased' : 'r', 'unbiased' : 'b'}[bias_type]
                linestyle = {key_ensemble : 'solid', key_oracle : 'dashed'}[prompt_type]
                test_set_str = {'biased' : 'biased test set', 'unbiased' : 'unbiased test set'}[bias_type]
                prompt_str = {key_ensemble : 'ensembled prompt', key_oracle : key_oracle.replace('_', '-') + ' prompt'}[prompt_type]
                legend_str = test_set_str + ', ' + prompt_str
                ys = []
                best_x = None
                best_y = float('-inf')
                for x, acc_dict in zip(xs, acc_dicts):
                    acc_subdict = acc_dict[bias_type + '_acc_as_percentage']
                    if prompt_type == key_ensemble:
                        acc = acc_subdict[key_ensemble]
                    else:
                        acc = np.mean([acc_subdict[key_oracle][non_target] for non_target in sorted(acc_subdict[key_oracle].keys())])

                    ys.append(acc)
                    if acc > best_y:
                        best_y = acc
                        best_x = x

                if should_add_to_legend:
                    plt.plot(xs, ys, color=color, linestyle=linestyle, label=legend_str)
                else:
                    plt.plot(xs, ys, color=color, linestyle=linestyle)

                plt.scatter([best_x], [best_y], s=320, marker='*', color='gold')
                plt.text(best_x, best_y, '%.1f%%'%(best_y))

        should_add_to_legend = False

    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, 0.7 * box.width, box.height])
    plt.legend(framealpha=1, bbox_to_anchor=(1,0.5), loc='center left')
    plt.xlabel('epochs')
    plt.ylabel('accuracy (%)')
    plt.savefig(plot_filename)
    plt.clf()

def usage():
    print('Usage: python make_corrupted_cifar10_plots.py <experiment_dirs> <plot_filename> [<swap_class_and_domain>=False]')

if __name__ == '__main__':
    make_corrupted_cifar10_plots(*(sys.argv[1:]))
