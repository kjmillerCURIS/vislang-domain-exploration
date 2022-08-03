import os
import sys
import copy
import glob
from tqdm import tqdm

def load_class2words_dict(base_dir):
    class2words_dict = {}
    f = open(os.path.join(base_dir, 'words.txt'), 'r')
    for line in f:
        ss = line.rstrip('\n').split()
        my_class = ss[0]
        words = ' '.join(ss[1:]).split(', ')
        class2words_dict[my_class] = words

    f.close()

    return class2words_dict

#returns words_dict, class2words_dict, class2filenames_dict
#base_dir should be something like "blahblah/yadayada/ILSVRC2012_val"
#it should have a bunch of folders in it that start with "n"
#and also a file in it called "words.txt" which is gotten from https://github.com/seshuad/IMagenet/blob/master/tiny-imagenet-200/words.txt
#words_dict will map image basename to a LIST of words (yeah, I know...)
#class2words_dict will map class to list of words
#class2filenames_dict will map class to a list of filenames (full paths)
#this has only been tested on ImageNet validation data
def load_non_image_data(base_dir, image_classes_only=True):
    words_dict = {}
    class2filenames_dict = {}

    class2words_dict = load_class2words_dict(base_dir)
    my_dirs = sorted(glob.glob(os.path.join(base_dir, 'n*')))
    for my_dir in tqdm(my_dirs):
        my_class = os.path.basename(my_dir)
        class2filenames_dict[my_class] = []
        words = class2words_dict[my_class]
        images = sorted(glob.glob(os.path.join(my_dir, '*.*')))
        images = [os.path.abspath(image) for image in images]
        class2filenames_dict[my_class] = images
        for image in images:
            image_base = os.path.basename(image)
            words_dict[image_base] = copy.deepcopy(words) #just to be safe

    if image_classes_only:
        class2words_dict = {classID : class2words_dict[classID] for classID in sorted(class2filenames_dict.keys())}

    return words_dict, class2words_dict, class2filenames_dict

if __name__ == '__main__':
    words_dict, class2words_dict, class2filenames_dict = load_non_image_data(os.path.expanduser('~/data/vislang-domain-exploration-data/ILSVRC2012_val'))
    import pdb
    pdb.set_trace()
