import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from plot_and_table_utils import ORDERED_AUG_IDS, FACECOLOR

def make_linearprobe_plot(linearprobe_stats_dict_filename, robustness_stats_dict_filename, plot_filename):
    with open(linearprobe_stats_dict_filename, 'rb') as f:
        linearprobe_stats_dict = pickle.load(f)

    with open(robustness_stats_dict_filename, 'rb') as f:
        robustness_stats_dict = pickle.load(f)

    plt.clf()
    fig, ax = plt.subplots(figsize=(10,16), ncols=1, sharey=True, facecolor=FACECOLOR)
    fig.tight_layout()
    ypos = np.arange(len(ORDERED_AUG_IDS))
    bar_offset = 0.18
    bar_height = 0.3
    ax.barh(ypos - bar_offset, [robustness_stats_dict['zeroshot'][1]['primary'][augID]['acc_decrease_as_percentage'] for augID in ORDERED_AUG_IDS], bar_height, align='center', color='r', alpha=0.5, zorder=10, label='zero-shot')
    ax.barh(ypos + bar_offset, [linearprobe_stats_dict['noop'] - linearprobe_stats_dict[augID] for augID in ORDERED_AUG_IDS], bar_height, align='center', color='g', alpha=0.5, zorder=10, label='linear probe')
    ax.invert_xaxis()
    plt.gca().invert_yaxis()
    ylim = ax.get_ylim()
    start_acc = robustness_stats_dict['zeroshot'][1]['secondary']['unaugmented_acc_as_percentage']
    ax.plot([start_acc, start_acc], ylim, linestyle='--', color='r', linewidth=5)
    start_acc = linearprobe_stats_dict['noop']
    ax.plot([start_acc, start_acc], ylim, linestyle='--', color='g', linewidth=5)
    ax.set_ylim(ylim)
    ax.set_xlim((ax.get_xlim()[0], 0))
    ax.set(yticks=ypos, yticklabels=ORDERED_AUG_IDS)
    ax.yaxis.tick_left()
    ax.set_title('Top1 acc decrease by augmenting image', fontsize=18)
    ax.set_facecolor(FACECOLOR)
    ax.grid(visible=True, which='both', axis='both', color='dimgray')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set(fontsize=14)

    for label in ax.get_xticklabels():
        if '%' not in label.get_text():
            label.set_text(label.get_text() + '%')

    plt.legend()
    plt.subplots_adjust(top=0.85, bottom=0.1, left=0.28, right=0.95)
    plt.savefig(plot_filename, facecolor=FACECOLOR)
    plt.clf()

def usage():
    print('Usage: python make_linearprobe_plot.py <linearprobe_stats_dict_filename> <robustness_stats_dict_filename> <plot_filename>')

if __name__ == '__main__':
    make_linearprobe_plot(*(sys.argv[1:]))
