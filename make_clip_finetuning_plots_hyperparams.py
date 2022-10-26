import os
import sys
import glob
import pickle
from experiment_params.balance_params import grab_params
from experiment_params.param_utils import get_params_key
from clip_finetuning_plot_utils import make_plots

#make_plots(results_list, color_list, marker_list, linestyle_list, label_list, plot_prefix)

EXPERIMENT_DIR_GLOB_STR = '../vislang-domain-exploration-data/Experiments/experiment_*Normal*'
PLOT_PREFIX = '../vislang-domain-exploration-data/clip_finetuning_plots_hyperparams/clip_finetuning_plot_hyperparams'

COLOR_DICT = {5e-4 : 'r', 1e-4 : 'orange', 1e-5 : 'y', 1e-6 : 'g', 1e-7 : 'b', 5e-8 : 'm', 1e-8 : 'k'}
MARKER_DICT = {True : '+', False : 'o'}
LINESTYLE = '-'

#returns results, color, marker, linestyle, label
def get_stuff(experiment_dir):
    results_filename = os.path.join(experiment_dir, 'val_zeroshot_results.pkl')
    with open(results_filename, 'rb') as f:
        results = pickle.load(f)

    p = grab_params(get_params_key(experiment_dir))
    color = COLOR_DICT[p.clip_learning_rate]
    marker = MARKER_DICT[p.english_only]
    linestyle = LINESTYLE
    label = 'lr=' + '{:.2e}'.format(p.clip_learning_rate) + ', english_only=' + str(p.english_only)
    return results, color, marker, linestyle, label

def make_clip_finetuning_plots_hyperparams():
    experiment_dirs = sorted(glob.glob(EXPERIMENT_DIR_GLOB_STR))
    results_list, color_list, marker_list, linestyle_list, label_list = zip(*[get_stuff(experiment_dir) for experiment_dir in experiment_dirs])
    make_plots(results_list, color_list, marker_list, linestyle_list, label_list, PLOT_PREFIX)

if __name__ == '__main__':
    make_clip_finetuning_plots_hyperparams()
