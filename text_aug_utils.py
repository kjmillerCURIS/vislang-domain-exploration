import os
import sys

def generate_text_aug_dict():
    aug_dict = {}
    aug_dict['noop'] = ['a photo of a %s']
    aug_dict['make_background_white'] = ['a clipart image of a %s', 'a photo of a %s with a white background']
    aug_dict['make_background_black'] = ['a photo of a %s with a black background']
    aug_dict['make_background_blue'] = ['a photo of a %s with a blue background']
    aug_dict['sketchify'] = ['a drawing of a %s', 'a sketch of a %s']
    aug_dict['posterize'] = ['a poster of a %s', 'a cartoon of a %s']
    aug_dict['low_res'] = ['a low resolution image of a %s', 'a low quality image of a %s', 'a pixelated image of a %s']
    aug_dict['blur'] = ['a blurry photo of a %s', 'a foggy photo of a %s']
    aug_dict['foggy'] = ['a blurry photo of a %s', 'a foggy photo of a %s']
    aug_dict['grayscale'] = ['a black and white photo of a %s', 'a vintage photo of a %s']
    aug_dict['sepia'] = ['a sepia tone photo of a %s', 'a vintage photo of a %s']
    aug_dict['dim'] = ['a dim photo of a %s', 'an underexposed photo of a %s']
    aug_dict['brighten'] = ['a bright photo of a %s', 'an overexposed photo of a %s']
    aug_dict['closeup'] = ['a closeup photo of a %s']
    aug_dict['fisheye'] = ['a fisheye photo of a %s', 'a barrel distorted photo of a %s']
    aug_dict['upside_down'] = ['an upside-down photo of a %s']
    aug_dict['sideways'] = ['a sideways photo of a %s']
    aug_dict['tilt'] = ['a tilted photo of a %s']
    return aug_dict
