import os
import sys
import itertools
import pickle
import random
from tqdm import tqdm
from non_image_data_utils_corrupted_cifar10 import CLASS_NAMES
from general_aug_utils_corrupted_cifar10 import generate_aug_dict

NUM_SPLITS = {'easy_zeroshot' : 10, 'hard_zeroshot' : 10}
SPLIT_TEST_RATIO = {'easy_zeroshot' : 0.5, 'hard_zeroshot' : 0.5}
SEED_FN = lambda split_index : split_index

def get_permutation(classes, domains):
    assert(len(classes) == len(domains))
    return list(zip(classes, random.sample(domains, len(domains))))

def setup_counter(key_list):
    return {k : 0 for k in key_list}

def check_easy_zeroshot_split(split):
    classes = sorted(CLASS_NAMES.keys())
    domains = sorted(generate_aug_dict().keys())
    assert(len(classes) == len(domains))
    test_target = int(round(SPLIT_TEST_RATIO['easy_zeroshot'] * len(classes)))
    train_target = len(classes) - test_target
    train_class_counts = setup_counter(classes)
    test_class_counts = setup_counter(classes)
    train_domain_counts = setup_counter(domains)
    test_domain_counts = setup_counter(domains)
    for (classID, domain) in split['train']:
        train_class_counts[classID] += 1
        train_domain_counts[domain] += 1
    
    for (classID, domain) in split['test']:
        test_class_counts[classID] += 1
        test_domain_counts[domain] += 1

    assert(all([train_class_counts[k] == train_target for k in classes]))
    assert(all([test_class_counts[k] == test_target for k in classes]))
    assert(all([train_domain_counts[k] == train_target for k in domains]))
    assert(all([test_domain_counts[k] == test_target for k in domains]))

def make_easy_zeroshot_split(random_seed):
    classes = sorted(CLASS_NAMES.keys())
    domains = sorted(generate_aug_dict().keys())
    assert(len(classes) == len(domains))
    random.seed(random_seed)
    train_groups = set([])
    num_perms = len(classes) - int(round(SPLIT_TEST_RATIO['easy_zeroshot'] * len(classes)))
    for t in range(num_perms):
        while True:
            perm_groups = get_permutation(classes, domains)
            if all([g not in train_groups for g in perm_groups]):
                train_groups.update(perm_groups)
                break

    test_groups = [g for g in itertools.product(classes, domains) if g not in train_groups]
    train_groups = sorted(train_groups)
    test_groups = sorted(test_groups)
    return {'train' : train_groups, 'test' : test_groups}

def make_hard_zeroshot_split(random_seed):
    classes = sorted(CLASS_NAMES.keys())
    domains = sorted(generate_aug_dict().keys())
    assert(len(classes) == len(domains))
    random.seed(random_seed)
    num_train = len(classes) - int(round(SPLIT_TEST_RATIO['hard_zeroshot'] * len(classes)))
    train_classes = random.sample(classes, num_train)
    train_domains = random.sample(domains, num_train)
    test_classes = [classID for classID in classes if classID not in train_classes]
    test_domains = [domain for domain in domains if domain not in train_domains]
    train_groups = sorted(itertools.product(train_classes, train_domains))
    test_groups = sorted(itertools.product(test_classes, test_domains))
    return {'train' : train_groups, 'test' : test_groups}

def make_trivial_split():
    classes = sorted(CLASS_NAMES.keys())
    domains = sorted(generate_aug_dict().keys())
    assert(len(classes) == len(domains))
    train_groups = sorted(itertools.product(classes, domains))
    test_groups = sorted(itertools.product(classes, domains))
    return {'train' : train_groups, 'test' : test_groups}

def make_corrupted_cifar10_group_splits(splits_filename):
    splits = {}
    splits['trivial'] = make_trivial_split()
    splits['easy_zeroshot'] = []
    for split_index in tqdm(range(NUM_SPLITS['easy_zeroshot'])):
        splits['easy_zeroshot'].append(make_easy_zeroshot_split(SEED_FN(split_index)))

    splits['hard_zeroshot'] = []
    for split_index in tqdm(range(NUM_SPLITS['hard_zeroshot'])):
        splits['hard_zeroshot'].append(make_hard_zeroshot_split(SEED_FN(split_index)))

    with open(splits_filename, 'wb') as f:
        pickle.dump(splits, f)

def usage():
    print('Usage: python make_corrupted_cifar10_group_splits.py <splits_filename>')

if __name__ == '__main__':
    make_corrupted_cifar10_group_splits(*(sys.argv[1:]))
