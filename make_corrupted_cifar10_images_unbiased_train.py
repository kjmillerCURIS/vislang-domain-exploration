import os
import sys
import random
from make_corrupted_cifar10_images import load_images_and_classes, assign_images_to_domains_one_class_test, apply_corruptions_and_save, ALL_DOMAINS, RANDOM_SEED

def make_corrupted_cifar10_images_unbiased_train(cifar10_dir, corrupted_cifar10_dir):
    random.seed(RANDOM_SEED)
    src_dir = os.path.join(cifar10_dir, 'train')
    dst_dir = os.path.join(corrupted_cifar10_dir, 'unbiased_train')
    image_buckets, class_domain_dict = load_images_and_classes(src_dir)
    for my_class in sorted(image_buckets.keys()):
        image_bases = image_buckets[my_class]
        domain_dict = assign_images_to_domains_one_class_test(image_bases, ALL_DOMAINS)
        for image_base in image_bases:
            class_domain_dict[image_base]['domain'] = domain_dict[image_base]

    apply_corruptions_and_save(src_dir, dst_dir, class_domain_dict)

def usage():
    print('Usage: python make_corrupted_cifar10_images_unbiased_train.py <cifar10_dir> <corrupted_cifar10_dir>')

if __name__ == '__main__':
    make_corrupted_cifar10_images_unbiased_train(*(sys.argv[1:]))
