import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from general_aug_utils import generate_aug_dict

ZOOM_DROP_THRESHOLD = 5.0
X_BUFFER = 0.005
Y_BUFFER = 1.0

def compute_epoch(result):
    return result['epoch'] + result['step_within_epoch'] / result['epoch_length']

def compute_avg_acc(result, standard_or_own_domain):
    d = result['zeroshot_top1_acc_as_percentage'][standard_or_own_domain + '_text_template']
    return np.mean([d[k] for k in sorted(d.keys())])

#returns epochs_list, result_list, zoom_index
def grab_data(results):
    epoch_result_pairs = sorted([(compute_epoch(results[k]), results[k]) for k in sorted(results.keys())])
    epoch_list, result_list = list(zip(*epoch_result_pairs)) #a pair of lists instead of a list of pairs
    acc_list = [compute_avg_acc(result, 'standard') for result in result_list]
    zoom_index = len(acc_list)
    for i in range(1, len(acc_list)):
        drop = acc_list[i-1] - acc_list[i]
        if drop > ZOOM_DROP_THRESHOLD:
            zoom_index = i
            break

    return epoch_list, result_list, zoom_index

def make_clip_finetuning_plots_helper(results, standard_or_own_domain, plot_prefix):
    epoch_list, result_list, zoom_index = grab_data(results)
    print('mrow')
    aug_IDs = sorted(generate_aug_dict().keys())
    avg_acc_list = [compute_avg_acc(result, standard_or_own_domain) for result in result_list]

    #zoomed out plot
    plt.clf()
    plt.plot(epoch_list, avg_acc_list, marker='o', label='finetuned')
    plt.plot(plt.xlim(), [avg_acc_list[0], avg_acc_list[0]], linestyle='--', label='OpenAI checkpoint')
    plt.xlabel('epoch')
    plt.ylabel('zero-shot acc (%) on augmented ImageNet1K val')
    plt.title(os.path.basename(plot_prefix) + ' ' + standard_or_own_domain + '_text_template')
    plt.legend()
    plt.xlim((0, plt.xlim()[1]))
    plt.savefig(plot_prefix + '-' + standard_or_own_domain + '-zoomedout.png')

    #zoomed in plot
    plt.xlim((0, epoch_list[zoom_index - 1] + X_BUFFER))
    plt.ylim((min(avg_acc_list[:zoom_index]) - Y_BUFFER, max(avg_acc_list[:zoom_index]) + Y_BUFFER))
    plt.savefig(plot_prefix + '-' + standard_or_own_domain + '-zoomedin.png')
    plt.clf()

    #multidomain plot (zoomed in)
    plt.clf()
    delta_lists = []
    for aug_ID in aug_IDs:
        delta_list = []
        baseline = result_list[0]['zeroshot_top1_acc_as_percentage'][standard_or_own_domain + '_text_template'][aug_ID]
        for result in result_list:
            d = result['zeroshot_top1_acc_as_percentage'][standard_or_own_domain + '_text_template']
            delta_list.append(d[aug_ID] - baseline)

        delta_lists.append(delta_list)

    for delta_list in delta_lists:
        plt.plot(epoch_list, delta_list, marker='o')
        plt.plot(plt.xlim(), [0,0], linestyle='--')

    plt.xlabel('epoch')
    plt.ylabel('acc change vs OpenAI checkpoint')
    plt.title(os.path.basename(plot_prefix) + ' ' + standard_or_own_domain + '_text_template')
    plt.xlim((0, epoch_list[zoom_index - 1] + X_BUFFER))
    ylim = (np.amin(np.array(delta_lists)[:,:zoom_index]) - Y_BUFFER, np.amax(np.array(delta_lists)[:,:zoom_index]) + Y_BUFFER)
    plt.ylim(ylim)
    plt.savefig(plot_prefix + '-' + standard_or_own_domain + '-multidomain.png')
    plt.clf()

def make_clip_finetuning_plots(experiment_dir, plot_prefix):
    experiment_dir = os.path.abspath(os.path.expanduser(experiment_dir))
    plot_prefix = os.path.abspath(os.path.expanduser(plot_prefix))
    os.makedirs(os.path.dirname(plot_prefix), exist_ok=True)
    with open(os.path.join(experiment_dir, 'val_zeroshot_results.pkl'), 'rb') as f:
        results = pickle.load(f)

    for standard_or_own_domain in ['standard', 'own_domain']:
        print('meow')
        make_clip_finetuning_plots_helper(results, standard_or_own_domain, plot_prefix)

def usage():
    print('Usage: python make_clip_finetuning_plots.py <experiment_dir> <plot_prefix>')

if __name__ == '__main__':
    make_clip_finetuning_plots(*(sys.argv[1:]))
