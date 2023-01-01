import os
import sys
import pickle
from non_image_data_utils_corrupted_cifar10 import CLASS_NAMES

def extract_bias_alignment(class_domain_dict_filename):
    with open(class_domain_dict_filename, 'rb') as f:
        class_domain_dict = pickle.load(f)

    bias_aligned_groups, group_freq_list = extract_bias_alignment_helper(class_domain_dict)
    print('Group frequencies: ' + str(group_freq_list))
    print('Bias-aligned groups:')
    print([(CLASS_NAMES[g[0]], g[1]) for g in bias_aligned_groups])
    print(len(bias_aligned_groups)) #you can visually check that this is 10 for CIFAR10

def extract_bias_alignment_helper(class_domain_dict):
    group_frequencies = {}
    for image_base in sorted(class_domain_dict.keys()):
        classID, domain = class_domain_dict[image_base]['class'], class_domain_dict[image_base]['domain']
        if (classID, domain) not in group_frequencies:
            group_frequencies[(classID, domain)] = 0

        group_frequencies[(classID, domain)] = group_frequencies[(classID, domain)] + 1

    frequency2groups = {}
    for k in sorted(group_frequencies.keys()):
        freq = group_frequencies[k]
        if freq not in frequency2groups:
            frequency2groups[freq] = []

        frequency2groups[freq].append(k)

    assert(len(frequency2groups.keys()) == 2)
    group_freq_list = sorted(frequency2groups.keys())
    bias_aligned_groups = sorted(frequency2groups[group_freq_list[1]])

    #now check that there's a 1-to-1 mapping by checking for duplicates
    classes = set([])
    domains = set([])
    for g in bias_aligned_groups:
        assert(g[0] not in classes)
        assert(g[1] not in domains)
        classes.add(g[0])
        domains.add(g[1])

    return bias_aligned_groups, group_freq_list

def usage():
    print('Usage: python extract_bias_alignment.py <class_domain_dict_filename>')

if __name__ == '__main__':
    extract_bias_alignment(*(sys.argv[1:]))
