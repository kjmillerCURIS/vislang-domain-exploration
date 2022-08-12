import os
import sys
import numpy as np
import pickle
from matplotlib import pyplot as plt
from plot_and_table_utils import FACECOLOR, NUM_EXPLAINED_VARIANCES_1D

def make_subspace_plot_one(my_dict, class_or_aug, plot_filename):
    plt.clf()
    plt.figure(figsize=(12,4))
    x = np.arange(1, NUM_EXPLAINED_VARIANCES_1D+1)
    y = np.square(my_dict[class_or_aug + '_comp_PCA']['explained_SDs'])
    y = 100.0 * y / np.sum(y)
    y = y[:NUM_EXPLAINED_VARIANCES_1D]
    ax1 = plt.subplot()
    l1 = ax1.scatter(x, y, marker='*', color='k')
    ax1.set_xlabel('PC #')
    ax1.set_ylabel('Explained Var Percentage')
    ax1.set_title('%s subspace PCs ranked by explained variance percentage'%(class_or_aug.capitalize()))
    ax1.set_xticks(ticks=x, labels=['%d'%(xx) for xx in x])
    ax2 = ax1.twinx()
    ax2.set_ylabel('PC singular values (absolute)')
    l2 = ax2.scatter(x, my_dict[class_or_aug + '_comp_PCA']['explained_SDs'][:NUM_EXPLAINED_VARIANCES_1D], marker='o', color='m')
    ax1.set_ylim((0, ax1.get_ylim()[1]))
    ax2.set_ylim((0, ax2.get_ylim()[1]))
    plt.legend([l1, l2], ['Explained Var Percentage', 'PC singular values'])
    plt.savefig(plot_filename)
    plt.clf()

def make_subspace_plots(stats_dict_filename, plot_prefix):
    os.makedirs(os.path.dirname(plot_prefix), exist_ok=True)

    with open(stats_dict_filename, 'rb') as f:
        stats_dict = pickle.load(f)

    for norm in ['unnormalized', 'normalized']:
        for embedding_type in ['image', 'text']:
            for class_or_aug in ['class', 'aug']:
                make_subspace_plot_one(stats_dict[norm][embedding_type], class_or_aug, plot_prefix + '-' + '_'.join([norm, embedding_type, class_or_aug]) + '_subspace_plot.png')

def usage():
    print('Usage: python make_subspace_plots.py <stats_dict_filename> <plot_prefix>')

if __name__ == '__main__':
    make_subspace_plots(*(sys.argv[1:]))
