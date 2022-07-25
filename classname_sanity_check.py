import os
import sys
import random
import shutil
from tqdm import tqdm
from non_image_data_utils import load_non_image_data

def prettify(word):
    #FIXME: learn to do it the right way, not the hacky way!
    special_chars = '.-_,\'"?!/#@%&^*()[]}{|+~`<>:;'
    for c in special_chars:
        word = word.replace(c, '')

    word = word.replace(' ', '_')

    return word

def classname_sanity_check(base_dir, num_classes, num_images_per_class, random_seed, dst_dir):
    base_dir = os.path.expanduser(base_dir)
    dst_dir = os.path.expanduser(dst_dir)
    num_classes = int(num_classes)
    num_images_per_class = int(num_images_per_class)
    os.makedirs(dst_dir, exist_ok=True)
    random.seed(random_seed)

    words_dict,class2words_dict,class2filenames_dict = load_non_image_data(os.path.expanduser('~/data/vislang-domain-exploration-data/ILSVRC2012_val'))
    my_classes = random.sample(sorted(class2filenames_dict.keys()), num_classes)
    for my_class in tqdm(my_classes):
        words = class2words_dict[my_class]
        words_str = '-'.join([prettify(word) for word in words])
        my_filenames = random.sample(class2filenames_dict[my_class], num_images_per_class)
        for i, my_filename in tqdm(enumerate(my_filenames)):
            dst_filename = os.path.join(dst_dir, words_str + '-%05d.JPEG'%(i))
            shutil.copy(my_filename, dst_filename)

def usage():
    print('Usage: python classname_sanity_check.py <base_dir> <num_classes> <num_images_per_class> <random_seed> <dst_dir>')

if __name__ == '__main__':
    classname_sanity_check(*(sys.argv[1:]))
