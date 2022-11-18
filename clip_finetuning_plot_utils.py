import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm

ZOOMIN_BUFFER = 1.0

def compute_epoch(result):
    return result['epoch'] + result['step_within_epoch'] / result['epoch_length']

def compute_avg_acc(result, standard_or_own_domain):
    d = result['zeroshot_top1_acc_as_percentage'][standard_or_own_domain + '_text_template']
    return np.mean([d[k] for k in sorted(d.keys())])

#returns epoch_list, acc_list
def grab_data(results, standard_or_own_domain):
    epoch_result_pairs = sorted([(compute_epoch(results[k]), results[k]) for k in sorted(results.keys())])
    epoch_list, result_list = list(zip(*epoch_result_pairs)) #a pair of lists instead of a list of pairs
    acc_list = [compute_avg_acc(result, standard_or_own_domain) for result in result_list]
    return epoch_list, acc_list

#will make 4 plots, toggling between zoomin vs zoomout and standard-prompt vs own-domain-prompt
#will put a gold star at the highest point of all the lines, with text of its y-value
#will put a grey dotted line at the starting value of the first sequence in results_list, without any label in the legend
#will try to put the legend below the plot
#zoomout will be organic. zoomin will just change ylim[0] to be the grey-line value minus some buffer
#will always plot accuracy as percentage, averaged across all domains
#will always plot x-axis as epochs
def make_plots(results_list, color_list, marker_list, linestyle_list, label_list, plot_prefix):
    os.makedirs(os.path.dirname(plot_prefix), exist_ok=True)
    for standard_or_own_domain in ['standard', 'own_domain']:
        plt.clf()
        plt.figure(figsize=[14.4, 4.8])
        best_x = None
        best_y = float('-inf')
        baseline_y = None
        for results, color, marker, linestyle, label in zip(results_list, color_list, marker_list, linestyle_list, label_list):
            epoch_list, acc_list = grab_data(results, standard_or_own_domain)
            plt.plot(epoch_list, acc_list, color=color, marker=marker, linestyle=linestyle, label=label)
            if baseline_y is None:
                baseline_y = acc_list[0]

            if max(acc_list) > best_y:
                best_y = max(acc_list)
                best_x = epoch_list[np.argmax(acc_list)]

        plt.plot(plt.xlim(), [baseline_y, baseline_y], linestyle='dashed', color='0.5')
        plt.scatter([best_x], [best_y], s=320, marker='*', color='gold')
        plt.text(best_x, best_y, '%.1f%%'%(best_y))
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, 0.7 * box.width, box.height])
        plt.legend(framealpha=1, bbox_to_anchor=(1,0.5), loc='center left')
        plt.title('(' + standard_or_own_domain + ' prompt)')
        plt.xlabel('epochs')
        plt.ylabel('zero-shot accuracy (%)')
        plt.savefig(plot_prefix + '-' + standard_or_own_domain + '-zoomout.png')
        plt.ylim((baseline_y - ZOOMIN_BUFFER, best_y + ZOOMIN_BUFFER))
        plt.savefig(plot_prefix + '-' + standard_or_own_domain + '-zoomin.png')
        plt.clf()
