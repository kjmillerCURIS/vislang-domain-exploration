import os
import sys
import glob
import numpy as np
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
from experiment_params.param_utils import get_train_type
from extract_bias_alignment import extract_bias_alignment_helper
from non_image_data_utils_corrupted_cifar10 import CLASS_NAMES
from make_corrupted_cifar10_plots import get_epoch_length

MIN_ACC = 0.0
MAX_ACC = 100.0

''' just make the individual slides for now, another script can make the videos '''

#returns epochs, pred_dicts (yes, this includes filtering for only integral epochs)
def get_pred_dicts(experiment_dir):
    result_filenames = sorted(glob.glob(os.path.join(experiment_dir, 'results', '*.pkl')))
    epoch_length = get_epoch_length(experiment_dir)
    pairs = []
    for result_filename in tqdm(result_filenames):
        with open(result_filename, 'rb') as f:
            result = pickle.load(f)

        x = result['epoch'] + result['step_within_epoch'] / epoch_length
        pairs.append((x, result['pred_dict']))

    pairs = sorted(pairs, key = lambda p: p[0])
    pairs = [p for p in pairs if p[0] == int(p[0])]
    epochs = [int(p[0]) for p in pairs]
    pred_dicts = [p[1] for p in pairs]
    return epochs, pred_dicts

#return group_accuracies as 2D npy array
def compute_group_accuracies(pred_dict, prompt_type, gt_class_domain_dict, classes, domains):
    accs = np.zeros((len(classes), len(domains)), np.float64)
    counts = np.zeros((len(classes), len(domains)), np.float64)
    if prompt_type == 'avg_domains':
        for image_base in sorted(pred_dict[prompt_type].keys()):
            classID = gt_class_domain_dict[image_base]['class']
            domain = gt_class_domain_dict[image_base]['domain']
            i = classes.index(classID)
            j = domains.index(domain)
            pred = pred_dict[prompt_type][image_base]
            correct = int(pred == classID)
            accs[i,j] += correct
            counts[i,j] += 1

    elif prompt_type == 'own_domain':
        for expected_domain in sorted(pred_dict[prompt_type].keys()):
            for image_base in sorted(pred_dict[prompt_type][expected_domain].keys()):
                classID = gt_class_domain_dict[image_base]['class']
                domain = gt_class_domain_dict[image_base]['domain']
                assert(domain == expected_domain)
                i = classes.index(classID)
                j = domains.index(domain)
                pred = pred_dict[prompt_type][expected_domain][image_base]
                correct = int(pred == classID)
                accs[i,j] += correct
                counts[i,j] += 1

    else:
        assert(False)

    assert(np.all(counts == counts[0,0]))
    group_accuracies = 100.0 * accs / counts
    return group_accuracies

def imshow_group_accuracies(group_accuracies, classes, domains, my_title, imshow_filename):
    assert(np.amin(group_accuracies) >= MIN_ACC)
    assert(np.amax(group_accuracies) <= MAX_ACC)
    plt.close()
    fig, ax = plt.subplots()
    ax.imshow(group_accuracies, vmin=MIN_ACC, vmax=MAX_ACC)
    ax.set_yticks(np.arange(len(classes)), labels=[CLASS_NAMES[classID] for classID in classes])
    ax.set_xticks(np.arange(len(domains)), labels=domains)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    for i in range(len(classes)):
        for j in range(len(domains)):
            ax.text(j, i, '%.1f'%(group_accuracies[i,j]), ha='center', va='center', color='w')
            ax.set_title(my_title)

    plt.savefig(imshow_filename)
    plt.close()

def animate_corrupted_cifar10_group_accuracies(experiment_dir, corrupted_cifar10_dir):
    #figure out what order to put classes and domains in your heatmaps
    train_type = get_train_type(experiment_dir)
    if train_type == 'unbiased_train': #just get same alignment as original training set
        alignment_class_domain_dict_filename = os.path.join(corrupted_cifar10_dir, 'train', 'class_domain_dict.pkl')
    else: #get it from the stats
        alignment_class_domain_dict_filename = os.path.join(corrupted_cifar10_dir, train_type, 'class_domain_dict.pkl')

    with open(alignment_class_domain_dict_filename, 'rb') as f:
        alignment_class_domain_dict = pickle.load(f)

    bias_aligned_groups, _ = extract_bias_alignment_helper(alignment_class_domain_dict)
    classes, domains = zip(*(bias_aligned_groups))

    gt_class_domain_dict_filename = os.path.join(corrupted_cifar10_dir, 'test', 'class_domain_dict.pkl')
    with open(gt_class_domain_dict_filename, 'rb') as f:
        gt_class_domain_dict = pickle.load(f)

    #now get preds
    epochs, pred_dicts = get_pred_dicts(experiment_dir)

    for prompt_type in ['avg_domains', 'own_domain']:
        frame_dir = os.path.join(experiment_dir, 'group_accuracies_animations', prompt_type)
        os.makedirs(frame_dir, exist_ok=True)
        for t, (epoch, pred_dict) in tqdm(enumerate(zip(epochs, pred_dicts))):
            group_accuracies = compute_group_accuracies(pred_dict, prompt_type, gt_class_domain_dict, classes, domains)
            my_title = 'prompt_type=%s, epoch=%d'%(prompt_type, epoch)
            imshow_filename = os.path.join(frame_dir, 'group_accuracies-' + prompt_type + '-frame%05d.png'%(t))
            imshow_group_accuracies(group_accuracies, classes, domains, my_title, imshow_filename)

def usage():
    print('Usage: python animate_corrupted_cifar10_group_accuracies.py <experiment_dir> <corrupted_cifar10_dir>')

if __name__ == '__main__':
    animate_corrupted_cifar10_group_accuracies(*(sys.argv[1:]))
