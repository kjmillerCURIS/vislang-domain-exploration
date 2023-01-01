import os
import sys
from clip_adapter_utils import grab_exposed_clip_backbones
from compute_CLIP_adapter_inputs_corrupted_cifar10 import CLIP_MODEL_TYPE, process_images

def compute_CLIP_adapter_inputs_corrupted_cifar10_unbiased_train(corrupted_cifar10_dir, adapter_input_dict_filename_prefix):
    exposed_image_backbone, _ = grab_exposed_clip_backbones(CLIP_MODEL_TYPE)
    process_images(os.path.join(corrupted_cifar10_dir, 'unbiased_train'), exposed_image_backbone, adapter_input_dict_filename_prefix + '-unbiased_train-images.pkl')

def usage():
    print('Usage: python compute_CLIP_adapter_inputs_corrupted_cifar10_unbiased_train.py <corrupted_cifar10_dir> <adapter_input_dict_filename_prefix>')

if __name__ == '__main__':
    compute_CLIP_adapter_inputs_corrupted_cifar10_unbiased_train(*(sys.argv[1:]))
