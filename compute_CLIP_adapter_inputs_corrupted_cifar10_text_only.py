import os
import sys
from clip_adapter_utils import grab_exposed_clip_backbones
from compute_CLIP_adapter_inputs_corrupted_cifar10 import CLIP_MODEL_TYPE, process_text

def compute_CLIP_adapter_inputs_corrupted_cifar10_text_only(adapter_input_dict_filename_prefix):
    _, exposed_text_backbone = grab_exposed_clip_backbones(CLIP_MODEL_TYPE)
    process_text(exposed_text_backbone, adapter_input_dict_filename_prefix + '-text.pkl')

def usage():
    print('Usage: python compute_CLIP_adapter_inputs_corrupted_cifar10_text_only.py <adapter_input_dict_filename_prefix>')

if __name__ == '__main__':
    compute_CLIP_adapter_inputs_corrupted_cifar10_text_only(*(sys.argv[1:]))
