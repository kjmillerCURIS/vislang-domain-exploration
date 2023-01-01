import os
import sys
import random
from make_corrupted_cifar10_images import make_corrupted_cifar10_images_train_or_test, DEFAULT_BIAS_CONFLICTING_PROP

def make_corrupted_cifar10_biased_train_images_custom_seed(cifar10_dir, corrupted_cifar10_dir, custom_seed, bias_conflicting_prop=DEFAULT_BIAS_CONFLICTING_PROP):
    custom_seed = int(custom_seed)
    assert(custom_seed != 0) #we already did this seed the very first time we sampled a training set
    random.seed(custom_seed)
    make_corrupted_cifar10_images_train_or_test(cifar10_dir, corrupted_cifar10_dir, bias_conflicting_prop, 'train', custom_dst_subdir='train_BiasSamplingSeed%d'%(custom_seed))

def usage():
    print('Usage: python make_corrupted_cifar10_biased_train_images_custom_seed.py <cifar10_dir> <corrupted_cifar10_dir> <custom_seed> [<bias_conflicting_prop>=DEFAULT_BIAS_CONFLICTING_PROP]')

if __name__ == '__main__':
    make_corrupted_cifar10_biased_train_images_custom_seed(*(sys.argv[1:]))
