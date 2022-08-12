import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from plot_and_table_utils import ORDERED_AUG_IDS, FACECOLOR

#got lots of help from https://sharkcoder.com/data-visualization/mpl-bidirectional
def make_cossim_decrease_and_recovery_plot(stats_dict, plot_filename):
    #stats_dict['cossim']['primary'][augID][k] where k is 'mean_decrease' or 'mean_recovery'
    plt.clf()
    fig, axs = plt.subplots(figsize=(16,16), ncols=2, sharey=True, facecolor=FACECOLOR)
    fig.tight_layout()
    ypos = np.arange(len(ORDERED_AUG_IDS))
    axs[0].barh(ypos, [stats_dict['cossim']['primary'][augID]['mean_decrease'] for augID in ORDERED_AUG_IDS], align='center', color='r', alpha=0.5, zorder=10)
    axs[1].barh(ypos, [stats_dict['cossim']['primary'][augID]['mean_recovery'] for augID in ORDERED_AUG_IDS], align='center', color='b', alpha=0.5, zorder=10)
    axs[0].barh(ypos, [max(-stats_dict['cossim']['primary'][augID]['mean_recovery'], 0) for augID in ORDERED_AUG_IDS], align='center', color='b', alpha=0.5, zorder=10)
    axs[1].barh(ypos, [max(-stats_dict['cossim']['primary'][augID]['mean_decrease'], 0) for augID in ORDERED_AUG_IDS], align='center', color='r', alpha=0.5, zorder=10)
    axs[0].invert_xaxis()
    plt.gca().invert_yaxis()
    ylim = axs[0].get_ylim()
    cossim_gap = stats_dict['cossim']['secondary']['avg_cossim_sameclass_unaug'] - stats_dict['cossim']['secondary']['avg_cossim_diffclass_unaug']
    axs[0].plot([cossim_gap, cossim_gap], ylim, linestyle='--', color='k', linewidth=5)
    axs[1].plot([cossim_gap, cossim_gap], ylim, linestyle='--', color='k', linewidth=5)
    axs[0].set_ylim(ylim)
    xmax = max(axs[0].get_xlim()[0], axs[1].get_xlim()[1])
    axs[0].set_xlim((xmax, 0))
    axs[1].set_xlim((0, xmax))
    axs[0].set(yticks=ypos, yticklabels=ORDERED_AUG_IDS)
    axs[0].yaxis.tick_left()
    axs[0].set_title('Mean cos-sim decrease by augmenting image', fontsize=18)
    axs[1].set_title('Mean cos-sim recovery by augmenting text', fontsize=18)
    axs[0].set_facecolor(FACECOLOR)
    axs[1].set_facecolor(FACECOLOR)
    axs[0].grid(visible=True, which='both', axis='both', color='dimgray')
    axs[1].grid(visible=True, which='both', axis='both', color='dimgray')
    for label in (axs[0].get_xticklabels() + axs[0].get_yticklabels() + axs[1].get_xticklabels() + axs[1].get_yticklabels()):
        label.set(fontsize=14)

    plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.18, right=0.95)
    plt.savefig(plot_filename, facecolor=FACECOLOR)
    plt.clf()

def make_zeroshot_decrease_plot(stats_dict, plot_filename):
    #stats_dict['zeroshot'][1]['primary'][augID]['acc_decrease_as_percentage']
    #stats_dict['zeroshot'][1]['secondary']['unaugmented_acc_as_percentage']
    plt.clf()
    fig, ax = plt.subplots(figsize=(10,16), ncols=1, sharey=True, facecolor=FACECOLOR)
    fig.tight_layout()
    ypos = np.arange(len(ORDERED_AUG_IDS))
    ax.barh(ypos, [stats_dict['zeroshot'][1]['primary'][augID]['acc_decrease_as_percentage'] for augID in ORDERED_AUG_IDS], align='center', color='r', alpha=0.5, zorder=10)
    ax.invert_xaxis()
    plt.gca().invert_yaxis()
    ylim = ax.get_ylim()
    start_acc = stats_dict['zeroshot'][1]['secondary']['unaugmented_acc_as_percentage']
    ax.plot([start_acc, start_acc], ylim, linestyle='--', color='k', linewidth=5)
    ax.set_ylim(ylim)
    ax.set_xlim((ax.get_xlim()[0], 0))
    ax.set(yticks=ypos, yticklabels=ORDERED_AUG_IDS)
    ax.yaxis.tick_left()
    ax.set_title('Top1 zero-shot acc decrease by augmenting image', fontsize=18)
    ax.set_facecolor(FACECOLOR)
    ax.grid(visible=True, which='both', axis='both', color='dimgray')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set(fontsize=14)

    for label in ax.get_xticklabels():
        if '%' not in label.get_text():
            label.set_text(label.get_text() + '%')

    plt.subplots_adjust(top=0.85, bottom=0.1, left=0.28, right=0.95)
    plt.savefig(plot_filename, facecolor=FACECOLOR)
    plt.clf()

def make_cossim_diffpair_and_samepair_recovery_plot(stats_dict, plot_filename):
    #stats_dict['cossim']['primary'][augID][k] where k is 'mean_recovery_diffclass' or 'mean_recovery'
    plt.clf()
    fig, axs = plt.subplots(figsize=(16,16), ncols=2, sharey=True, facecolor=FACECOLOR)
    fig.tight_layout()
    ypos = np.arange(len(ORDERED_AUG_IDS))
    axs[0].barh(ypos, [stats_dict['cossim']['primary'][augID]['mean_recovery_diffclass'] for augID in ORDERED_AUG_IDS], align='center', color='orange', alpha=0.5, zorder=10)
    axs[1].barh(ypos, [stats_dict['cossim']['primary'][augID]['mean_recovery'] for augID in ORDERED_AUG_IDS], align='center', color='b', alpha=0.5, zorder=10)
    axs[0].barh(ypos, [max(-stats_dict['cossim']['primary'][augID]['mean_recovery'], 0) for augID in ORDERED_AUG_IDS], align='center', color='b', alpha=0.5, zorder=10)
    axs[1].barh(ypos, [max(-stats_dict['cossim']['primary'][augID]['mean_recovery_diffclass'], 0) for augID in ORDERED_AUG_IDS], align='center', color='orange', alpha=0.5, zorder=10)
    axs[0].invert_xaxis()
    plt.gca().invert_yaxis()
    ylim = axs[0].get_ylim()
    cossim_gap = stats_dict['cossim']['secondary']['avg_cossim_sameclass_unaug'] - stats_dict['cossim']['secondary']['avg_cossim_diffclass_unaug']
    axs[0].plot([cossim_gap, cossim_gap], ylim, linestyle='--', color='k', linewidth=5)
    axs[1].plot([cossim_gap, cossim_gap], ylim, linestyle='--', color='k', linewidth=5)
    axs[0].set_ylim(ylim)
    xmax = max(axs[0].get_xlim()[0], axs[1].get_xlim()[1])
    axs[0].set_xlim((xmax, 0))
    axs[1].set_xlim((0, xmax))
    axs[0].set(yticks=ypos, yticklabels=ORDERED_AUG_IDS)
    axs[0].yaxis.tick_left()
    axs[0].set_title('Mean cos-sim recovery (diff-class)', fontsize=18)
    axs[1].set_title('Mean cos-sim recovery (same-class)', fontsize=18)
    axs[0].set_facecolor(FACECOLOR)
    axs[1].set_facecolor(FACECOLOR)
    axs[0].grid(visible=True, which='both', axis='both', color='dimgray')
    axs[1].grid(visible=True, which='both', axis='both', color='dimgray')
    for label in (axs[0].get_xticklabels() + axs[0].get_yticklabels() + axs[1].get_xticklabels() + axs[1].get_yticklabels()):
        label.set(fontsize=14)

    plt.subplots_adjust(wspace=0, top=0.85, bottom=0.1, left=0.18, right=0.95)
    plt.savefig(plot_filename, facecolor=FACECOLOR)
    plt.clf()

def make_robustness_plots(stats_dict_filename, plot_prefix):
    os.makedirs(os.path.dirname(plot_prefix), exist_ok=True)

    with open(stats_dict_filename, 'rb') as f:
        stats_dict = pickle.load(f)

    make_cossim_decrease_and_recovery_plot(stats_dict, plot_prefix + '-cossim_decrease_and_recovery_plot.png')
    make_zeroshot_decrease_plot(stats_dict, plot_prefix + '-zeroshot_decrease_plot.png')
    make_cossim_diffpair_and_samepair_recovery_plot(stats_dict, plot_prefix + '-cossim_diffpair_and_samepair_recovery_plot.png')

def usage():
    print('Usage: python make_robustness_plots.py <stats_dict_filename> <plot_prefix>')

if __name__ == '__main__':
    make_robustness_plots(*(sys.argv[1:]))
