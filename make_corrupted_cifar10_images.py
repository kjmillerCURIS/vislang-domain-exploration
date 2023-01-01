import os
import sys
import copy
import cv2
import glob
import numpy as np
import pickle
import random
from tqdm import tqdm
from general_aug_utils_corrupted_cifar10 import generate_aug_dict

ALL_DOMAINS = ['Snow', 'Frost', 'Fog', 'Brightness', 'Contrast', 'Spatter', 'Elastic', 'JPEG', 'Pixelate', 'Saturate']
DEFAULT_BIAS_CONFLICTING_PROP = 0.0468
RANDOM_SEED = 0 #for mapping classes to domains, and for assigning images within a class to their respective domains

def assign_images_to_domains_one_class_train(image_bases, all_domains, preferred_domain, bias_conflicting_prop):
    num_bias_aligned = (1.0 - bias_conflicting_prop) * len(image_bases)
    assert(num_bias_aligned == int(round(num_bias_aligned)))
    num_bias_aligned = int(round(num_bias_aligned))
    num_bias_conflicting_per_domain = bias_conflicting_prop * len(image_bases) / (1.0 * len(all_domains) - 1)
    assert(num_bias_conflicting_per_domain == int(round(num_bias_conflicting_per_domain)))
    num_bias_conflicting_per_domain = int(round(num_bias_conflicting_per_domain))
    domain_dict = {}
    bias_aligned_image_bases = random.sample(image_bases, num_bias_aligned)
    for image_base in bias_aligned_image_bases:
        domain_dict[image_base] = preferred_domain

    bias_conflicting_image_bases = copy.deepcopy([image_base for image_base in image_bases if image_base not in bias_aligned_image_bases])
    random.shuffle(bias_conflicting_image_bases)
    non_preferred_domains = [domain for domain in all_domains if domain != preferred_domain]
    for i, image_base in enumerate(bias_conflicting_image_bases):
        domain_dict[image_base] = non_preferred_domains[i // num_bias_conflicting_per_domain]

    return domain_dict

def assign_images_to_domains_one_class_test(image_bases, all_domains):
    num_per_domain = len(image_bases) / (1.0 * len(all_domains))
    assert(num_per_domain == int(round(num_per_domain)))
    num_per_domain = int(round(num_per_domain))
    image_bases = copy.deepcopy(image_bases)
    random.shuffle(image_bases)
    domain_dict = {}
    for i, image_base in enumerate(image_bases):
        domain_dict[image_base] = all_domains[i // num_per_domain]

    return domain_dict

#returns:
#-image-bases bucketed by class
#-class_dict
def load_images_and_classes(train_or_test_dir):
    image_bases = sorted(glob.glob(os.path.join(train_or_test_dir, 'images', '*.png')))
    image_bases = [os.path.basename(x) for x in image_bases]
    with open(os.path.join(train_or_test_dir, 'class_dict.pkl'), 'rb') as f:
        class_dict = pickle.load(f)

    image_buckets = {}
    for image_base in image_bases:
        my_class = class_dict[image_base]['class']
        if my_class not in image_buckets:
            image_buckets[my_class] = []

        image_buckets[my_class].append(image_base)

    return image_buckets, class_dict

#class_domain_dict should map each image_base to a dict with keys 'class' and 'domain'
#this method will load, corrupt, and save the images, and will also update their names in the class_domain_dict and save that too
def apply_corruptions_and_save(src_train_or_test_dir, dst_train_or_test_dir, class_domain_dict):
    os.makedirs(dst_train_or_test_dir, exist_ok=True)
    os.makedirs(os.path.join(dst_train_or_test_dir, 'images'), exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(src_train_or_test_dir, 'images', '*.png')))
    dst_class_domain_dict = {}
    aug_dict = generate_aug_dict()
    for image_path in image_paths:
        image_base = os.path.basename(image_path)
        domain = class_domain_dict[image_base]['domain']
        aug_fn = aug_dict[domain]['image_aug_fn']
        numI = cv2.imread(image_path)
        numIaug = aug_fn(numI)
        assert(numIaug.shape[2] == 3)
        image_base_dst = os.path.splitext(image_base)[0] + '_' + domain + '.png'
        dst_class_domain_dict[image_base_dst] = class_domain_dict[image_base]
        cv2.imwrite(os.path.join(dst_train_or_test_dir, 'images', image_base_dst), numIaug)

    with open(os.path.join(dst_train_or_test_dir, 'class_domain_dict.pkl'), 'wb') as f:
        pickle.dump(dst_class_domain_dict, f)

def make_corrupted_cifar10_images_train_or_test(cifar10_dir, corrupted_cifar10_dir, bias_conflicting_prop, train_or_test, custom_dst_subdir=None):
    assert(train_or_test in ['train', 'test'])
    src_train_or_test_dir = os.path.join(cifar10_dir, train_or_test)
    dst_subdir = train_or_test
    if custom_dst_subdir is not None:
        dst_subdir = custom_dst_subdir

    dst_train_or_test_dir = os.path.join(corrupted_cifar10_dir, dst_subdir)
    image_buckets, class_domain_dict = load_images_and_classes(src_train_or_test_dir)
    if train_or_test == 'test':
        for my_class in sorted(image_buckets.keys()):
            image_bases = image_buckets[my_class]
            domain_dict = assign_images_to_domains_one_class_test(image_bases, ALL_DOMAINS)
            for image_base in image_bases:
                class_domain_dict[image_base]['domain'] = domain_dict[image_base]

    elif train_or_test == 'train':
        all_classes = sorted(image_buckets.keys())
        assert(len(all_classes) == len(ALL_DOMAINS))
        shuffled_domains = copy.deepcopy(ALL_DOMAINS)
        random.shuffle(shuffled_domains)
        preferred_domains = {k : v for k, v in zip(all_classes, shuffled_domains)}
        for my_class in all_classes:
            image_bases = image_buckets[my_class]
            domain_dict = assign_images_to_domains_one_class_train(image_bases, ALL_DOMAINS, preferred_domains[my_class], bias_conflicting_prop)
            for image_base in image_bases:
                class_domain_dict[image_base]['domain'] = domain_dict[image_base]

    else:
        assert(False)

    apply_corruptions_and_save(src_train_or_test_dir, dst_train_or_test_dir, class_domain_dict)

#assumes that you already got the CIFAR10 images with extract_cifar10_images.py
#structure should be:
#-"train"/"images"
#-"train"/"class_dict" (will have singletons with key "class") (will become "class_domain_dict" and have keys "class" and "domain")
#-ditto for "test"
def make_corrupted_cifar10_images(cifar10_dir, corrupted_cifar10_dir, bias_conflicting_prop=DEFAULT_BIAS_CONFLICTING_PROP):
    random.seed(RANDOM_SEED)
    make_corrupted_cifar10_images_train_or_test(cifar10_dir, corrupted_cifar10_dir, bias_conflicting_prop, 'train')
    make_corrupted_cifar10_images_train_or_test(cifar10_dir, corrupted_cifar10_dir, bias_conflicting_prop, 'test')

def usage():
    print('Usage: python make_corrupted_cifar10_images.py <cifar10_dir> <corrupted_cifar10_dir> [<bias_conflicting_prop>=%f]'%(DEFAULT_BIAS_CONFLICTING_PROP))

if __name__ == '__main__':
    make_corrupted_cifar10_images(*(sys.argv[1:]))
