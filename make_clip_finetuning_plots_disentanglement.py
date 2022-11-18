import os
import sys
import glob
import pickle
from experiment_params.balance_params import grab_params
from experiment_params.param_utils import get_params_key
from clip_finetuning_plot_utils import make_plots

'''
Makes the following plots:
0. Plot the baseline results.
1. Plot all the disentanglement results together in one plot.
2. 0+1 in a single plot.
3. Go through each baseline hyperparam setting that has disentanglement counterparts, and plot those on separate plots.
That's it! If you want to do a by-domain comparison, you should just grab the "best" (by avg) checkpoint from each side and make a table!

Order of presentation will be 0 (both), 2 (both), 1 (standard), 3 (standard), (4 (standard))
'''

#make_plots(results_list, color_list, marker_list, linestyle_list, label_list, plot_prefix)

EXPERIMENT_DIR_GLOB_STR = '../vislang-domain-exploration-data/Experiments/experiment_*Normal*'
PLOT_DIR = '../vislang-domain-exploration-data/clip_finetuning_plots_disentanglement'

COLOR_DICT = {5e-4 : 'r', 1e-4 : 'orange', 1e-5 : 'y', 1e-6 : 'g', 1e-7 : 'b', 5e-8 : 'm', 1e-8 : 'k'}
MARKER_DICT = {True : '+', False : 'o'}
LINESTYLE_FN = lambda p: 'solid' if not p.clip_finetuning_do_disentanglement else {0.1 : 'dashed', 1.0 : 'dotted'}[p.disentanglement_lambda]

#returns results, color, marker, linestyle, label
def get_stuff(experiment_dir):
    results_filename = os.path.join(experiment_dir, 'val_zeroshot_results.pkl')
    with open(results_filename, 'rb') as f:
        results = pickle.load(f)

    p = grab_params(get_params_key(experiment_dir))
    color = COLOR_DICT[p.clip_learning_rate]
    marker = MARKER_DICT[p.english_only]
    linestyle = LINESTYLE_FN(p)
    label = 'lr=' + '{:.2e}'.format(p.clip_learning_rate) + ', english_only=' + str(p.english_only)
    if p.clip_finetuning_do_disentanglement:
        label += ', disentangle(lambda=%.1f)'%(p.disentanglement_lambda)

    return results, color, marker, linestyle, label

#all experiment dirs associated with baseline experiments
def grab_experiment_dirs_baseline():
    BASELINE_LRS = ['', 'LR4', 'LR5', 'LR6', 'LR7', 'LR5e8', 'LR8']
    ENGLISH_ONLYS = ['', 'EnglishOnly']
    EXPERIMENT_PREFIX = '../vislang-domain-exploration-data/Experiments/experiment_'
    experiment_dirs = []
    for baseline_lr in BASELINE_LRS:
        for english_only in ENGLISH_ONLYS:
            experiment_dirs.append(os.path.abspath(os.path.expanduser(EXPERIMENT_PREFIX + english_only + 'NormalBatchingParams' + baseline_lr)))

    return experiment_dirs

#all experiment dirs associated with disentanglement experiments
def grab_experiment_dirs_treatment():
    TREATMENT_LRS = ['LR7', 'LR5e8']
    ENGLISH_ONLYS = ['', 'EnglishOnly']
    LAMBDA_VALS = ['Lambda0_1', 'Lambda1_0']
    EXPERIMENT_PREFIX = '../vislang-domain-exploration-data/Experiments/experiment_'
    experiment_dirs = []
    for treatment_lr in TREATMENT_LRS:
        for english_only in ENGLISH_ONLYS:
            for lambda_val in LAMBDA_VALS:
                experiment_dirs.append(os.path.abspath(os.path.expanduser(EXPERIMENT_PREFIX + english_only + 'DisentanglementParams' + treatment_lr + lambda_val)))

    return experiment_dirs

#baseline + treatment
def grab_experiment_dirs_both():
    return grab_experiment_dirs_baseline() + grab_experiment_dirs_treatment()

#get bucket for each baseline hyperparam as long as there's a treatment
#returns dict mapping (lr, english_only) to list of experiment dirs
def grab_buckets_of_experiment_dirs():
    experiment_dirs = grab_experiment_dirs_both()
    buckets = {}
    has_treatment = set([])
    for experiment_dir in experiment_dirs:
        p = grab_params(get_params_key(experiment_dir))
        k = (p.clip_learning_rate, p.english_only)
        if p.clip_finetuning_do_disentanglement:
            has_treatment.add(k)

        if k not in buckets:
            buckets[k] = []

        buckets[k].append(experiment_dir)

    buckets = {k : buckets[k] for k in sorted(buckets.keys()) if k in has_treatment}
    return buckets

def make_clip_finetuning_plots_disentanglement():

    #0. Plot the baseline results.
    experiment_dirs = grab_experiment_dirs_baseline()
    results_list, color_list, marker_list, linestyle_list, label_list = zip(*[get_stuff(experiment_dir) for experiment_dir in experiment_dirs])
    make_plots(results_list, color_list, marker_list, linestyle_list, label_list, os.path.join(PLOT_DIR, 'baseline'))

    #1. Plot all the disentanglement results together in one plot.
    experiment_dirs = grab_experiment_dirs_treatment()
    results_list, color_list, marker_list, linestyle_list, label_list = zip(*[get_stuff(experiment_dir) for experiment_dir in experiment_dirs])
    make_plots(results_list, color_list, marker_list, linestyle_list, label_list, os.path.join(PLOT_DIR, 'disentanglement'))

    #2. 0+1 in a single plot.
    experiment_dirs = grab_experiment_dirs_both()
    results_list, color_list, marker_list, linestyle_list, label_list = zip(*[get_stuff(experiment_dir) for experiment_dir in experiment_dirs])
    make_plots(results_list, color_list, marker_list, linestyle_list, label_list, os.path.join(PLOT_DIR, 'both'))

    #3. Go through each baseline hyperparam setting that has disentanglement counterparts, and plot those on separate plots.
    buckets = grab_buckets_of_experiment_dirs()
    for k in sorted(buckets.keys()):
        experiment_dirs = buckets[k]
        results_list, color_list, marker_list, linestyle_list, label_list = zip(*[get_stuff(experiment_dir) for experiment_dir in experiment_dirs])
        make_plots(results_list, color_list, marker_list, linestyle_list, label_list, os.path.join(PLOT_DIR, 'bucket-lr' + '{:.2e}'.format(k[0]) + '-english_only_%s'%(str(k[1]))))

if __name__ == '__main__':
    make_clip_finetuning_plots_disentanglement()
