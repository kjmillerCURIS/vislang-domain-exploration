import os
import sys
from image_aug_utils import generate_image_aug_dict
from text_aug_utils import generate_text_aug_dict

def generate_aug_dict():
    image_aug_dict = generate_image_aug_dict()
    text_aug_dict = generate_text_aug_dict()
    assert(sorted(image_aug_dict.keys()) == sorted(text_aug_dict.keys()))
    aug_dict = {}
    for augID in sorted(image_aug_dict.keys()):
        aug_dict[augID] = {}
        aug_dict[augID]['image_aug_fn'] = image_aug_dict[augID]
        aug_dict[augID]['text_aug_templates'] = text_aug_dict[augID]

    return aug_dict

if __name__ == '__main__':
    aug_dict = generate_aug_dict()
    import pdb
    pdb.set_trace()
