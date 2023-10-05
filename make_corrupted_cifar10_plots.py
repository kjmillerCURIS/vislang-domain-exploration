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

def get_result_base_dirs(experiment_dir, split_type, swap_class_and_domain=False):
    if split_type == 'trivial':
        if swap_class_and_domain:
            return ['results-predict_domain']
        else:
            return ['results']
    else:
        assert(split_type in ['easy_zeroshot', 'hard_zeroshot'])
        if swap_class_and_domain:
            dirs = sorted(glob.glob(os.path.join(experiment_dir, 'results-%s-*-predict_domain'%(split_type))))
            dirs = [os.path.basename(my_dir) for my_dir in dirs]
            return dirs
        else:
            dirs = sorted(glob.glob(os.path.join(experiment_dir, 'results-%s-*'%(split_type))))
            dirs = [os.path.basename(my_dir) for my_dir in dirs if 'predict_domain' not in os.path.basename(my_dir)]
            return dirs

#experiment_dirs can be a comma-separated list, or just one dir, or it could be an actual list
def make_corrupted_cifar10_plots(experiment_dirs, plot_filename, swap_class_and_domain=False, split_type='trivial'):
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
    already_in_legend = set([])
    for experiment_dir in experiment_dirs:
        p = grab_params(get_params_key(experiment_dir))
        use_domainless = (not swap_class_and_domain) and (p.domainless_text_prop > 0.0)
        epoch_length = get_epoch_length(experiment_dir)
        result_base_dirs = get_result_base_dirs(experiment_dir, split_type, swap_class_and_domain=swap_class_and_domain)
        print('HI KEZVIN: %s'%(str(result_base_dirs)))
        xs_list = []
        acc_dicts_list = []
        for result_base_dir in result_base_dirs:
            result_filenames = sorted(glob.glob(os.path.join(experiment_dir, result_base_dir, '*.pkl')))
            print('HI KEZVIN: %s'%(str(result_filenames)))
            xs, acc_dicts = get_acc_dicts(result_filenames, epoch_length)
            xs_list.append(xs)
            acc_dicts_list.append(acc_dicts)

        for bias_type in ['biased', 'unbiased']:
            prompt_types = [key_ensemble, key_oracle]
            if use_domainless:
                prompt_types.extend(['domainless', 'avg_domains_plus_domainless', 'own_domain_plus_domainless'])

            for prompt_type in prompt_types:
                color = {'biased' : 'r', 'unbiased' : 'b'}[bias_type]
                if prompt_type == 'domainless':
                    color = {'r' : 'orange', 'b' : 'g'}[color]
                elif prompt_type in ['avg_domains_plus_domainless', 'own_domain_plus_domainless']:
                    color = {'r' : 'pink', 'b' : 'deepskyblue'}[color]

                linestyle = {key_ensemble : 'solid', key_oracle : 'dashed', 'domainless' : 'solid', 'avg_domains_plus_domainless' : 'solid', 'own_domain_plus_domainless' : 'dashed'}[prompt_type]
                test_set_str = {'biased' : 'biased test set', 'unbiased' : 'unbiased test set'}[bias_type]
                prompt_str = {key_ensemble : 'ensembled prompt', key_oracle : key_oracle.replace('_', '-') + ' prompt', 'domainless' : 'domainless prompt', 'avg_domains_plus_domainless' : 'ensembled+domainless prompt', 'own_domain_plus_domainless' : 'own-domain+domainless prompt'}[prompt_type]
                legend_str = test_set_str + ', ' + prompt_str
                plot_xs = xs_list[0]
                plot_ys = []
                best_x = None
                best_y = float('-inf')
                assert(all([len(xs) == len(xs_list[0]) for xs in xs_list]))
                print('HI KEZVIN: %d'%(len(xs_list[0])))
                for index in range(len(xs_list[0])):
                    acc_list = []
                    assert(all([xs[index] == xs_list[0][index] for xs in xs_list]))
                    for xs, acc_dicts in zip(xs_list, acc_dicts_list):
                        x = xs[index]
                        acc_dict = acc_dicts[index]
                        acc_subdict = acc_dict[bias_type + '_acc_as_percentage']
                        if prompt_type in [key_ensemble, 'domainless', 'avg_domains_plus_domainless']:
                            acc = acc_subdict[prompt_type]
                        else:
                            total_oracle_weight = np.sum([acc_subdict['oracle_weights'][non_target] for non_target in sorted(acc_subdict['oracle_weights'].keys())])
                            acc = np.sum([acc_subdict[prompt_type][non_target] * acc_subdict['oracle_weights'][non_target] for non_target in sorted(acc_subdict[prompt_type].keys())])
                            acc /= total_oracle_weight

                        acc_list.append(acc)

                    avg_acc = np.mean(acc_list)
                    plot_ys.append(avg_acc)
                    if avg_acc > best_y:
                        best_y = avg_acc
                        best_x = x

                if (bias_type, prompt_type) not in already_in_legend:
                    plt.plot(plot_xs, plot_ys, color=color, linestyle=linestyle, label=legend_str)
                else:
                    plt.plot(plot_xs, plot_ys, color=color, linestyle=linestyle)

                plt.scatter([best_x], [best_y], s=320, marker='*', color='gold')
                plt.text(best_x, best_y, '%.1f%%'%(best_y))
                already_in_legend.add((bias_type, prompt_type))

    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, 0.7 * box.width, box.height])
    plt.legend(framealpha=1, bbox_to_anchor=(1,0.5), loc='center left')
    plt.xlabel('epochs')
    plt.ylabel('accuracy (%)')
    plt.savefig(plot_filename)
    plt.clf()

def usage():
    print('Usage: python make_corrupted_cifar10_plots.py <experiment_dirs> <plot_filename> [<swap_class_and_domain>=False] [<split_type>="trivial"]')

if __name__ == '__main__':
    make_corrupted_cifar10_plots(*(sys.argv[1:]))
