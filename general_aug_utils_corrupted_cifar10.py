import os
import sys
import cv2
import numpy as np
import torchvision.transforms as trn
from cifar10_corruption_utils.make_cifar_c import snow, frost, fog, brightness, contrast, spatter, elastic_transform, jpeg_compression, pixelate, saturate

#['Snow', 'Frost', 'Fog', 'Brightness', 'Contrast', 'Spatter', 'Elastic', 'JPEG', 'Pixelate', 'Saturate']

CORRUPTION_SEVERITY = 4 #this is what https://arxiv.org/pdf/2007.02561.pdf used for their main result 

convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])

#this *shouldn't* be necessary if make_cifar_c actually worked as advertised, but it seems that half of the functions just return PIL.Image instead of np.ndarray
def convert_to_npy_if_necessary(x):
    if not isinstance(x, np.ndarray):
        return np.array(x)[:,:,:3]
    else:
        return x

def make_aug_target(base_fn):
    return {'image_aug_fn' : lambda numI: np.ascontiguousarray(np.uint8(convert_to_npy_if_necessary(base_fn(convert_img(np.ascontiguousarray(numI[:,:,::-1])), severity=CORRUPTION_SEVERITY))[:,:,::-1]))}

def generate_image_aug_fn_dict():
    aug_dict = {}
    aug_dict['Snow'] = make_aug_target(snow)
    aug_dict['Frost'] = make_aug_target(frost)
    aug_dict['Fog'] = make_aug_target(fog)
    aug_dict['Brightness'] = make_aug_target(brightness)
    aug_dict['Contrast'] = make_aug_target(contrast)
    aug_dict['Spatter'] = make_aug_target(spatter)
    aug_dict['Elastic'] = make_aug_target(elastic_transform)
    aug_dict['JPEG'] = make_aug_target(jpeg_compression)
    aug_dict['Pixelate'] = make_aug_target(pixelate)
    aug_dict['Saturate'] = make_aug_target(saturate)
    return aug_dict

def generate_aug_dict():
    aug_dict = generate_image_aug_fn_dict()
    aug_dict['Snow']['text_aug_template'] = 'a snowy photo of a %s'
    aug_dict['Frost']['text_aug_template'] = 'a frosted photo of a %s'
    aug_dict['Fog']['text_aug_template'] = 'a foggy photo of a %s'
    aug_dict['Brightness']['text_aug_template'] = 'an overexposed photo of a %s'
    aug_dict['Contrast']['text_aug_template'] = 'a low-contrast photo of a %s'
    aug_dict['Spatter']['text_aug_template'] = 'a spattered photo of a %s'
    aug_dict['Elastic']['text_aug_template'] = 'a warped photo of a %s'
    aug_dict['JPEG']['text_aug_template'] = 'a compressed photo of a %s'
    aug_dict['Pixelate']['text_aug_template'] = 'a pixelated photo of a %s'
    aug_dict['Saturate']['text_aug_template'] = 'a color-saturated photo of a %s'
    return aug_dict

def generate_domainless_text_template():
    return 'a photo of a %s'

if __name__ == '__main__':
    aug_dict = generate_aug_dict()
    import pdb
    pdb.set_trace()
