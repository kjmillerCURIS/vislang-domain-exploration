import os
import sys
import glob
import numpy as np
import pickle
from PIL import Image
import torch
from tqdm import tqdm
import clip
from download_images_from_laion import get_domain_dir
from experiment_params.param_utils import get_params_key
from experiment_params.balance_params import grab_params

class LaionImageAndTextDatasetOneDomain(torch.utils.data.Dataset):

    def __make_image_preprocessor(self, clip_model_type):
        _, self.preprocess = clip.load(clip_model_type, device='cpu')

    def __get_image_filename(self, t):
        num_str = '%09d'%(t)
        dir_name = num_str[:5]
        jpg_base = num_str + '.jpg'
        return os.path.join(self.domain_dir, 'images', dir_name, jpg_base)

    #this assumes that self.domain_dir has already been set
    def __load_image_filename_list_and_text_str_list(self):
        self.image_filename_list = []
        self.text_str_list = []
        with open(os.path.join(self.domain_dir, 'image_bases.pkl'), 'rb') as f:
            image_bases = pickle.load(f)

        with open(os.path.join(self.domain_dir, 'image_base_to_caption.pkl'), 'rb') as f:
            image_base_to_caption = pickle.load(f)

        for t, image_base in enumerate(image_bases):
            image_filename = self.__get_image_filename(t)
            if not os.path.exists(image_filename):
                continue

            self.image_filename_list.append(image_filename)
            self.text_str_list.append(image_base_to_caption[image_base])

    ''' Use this to fine-tune CLIP on image-text pairs of a subset of laion (just one "domain") '''
    def __init__(self, experiment_dir, domain_index):
        self.domain_dir = get_domain_dir(experiment_dir, domain_index)
        params_key = get_params_key(experiment_dir)
        p = grab_params(params_key)
        self.__make_image_preprocessor(p.clip_model_type)
        self.__load_image_filename_list_and_text_str_list()

    #if image is missing, we'll move forward until we find an index that has an image
    #use fp16 for the image for now
    def __getitem__(self, idx):
        return {'image' : self.preprocess(Image.open(self.image_filename_list[idx])).type(torch.float16), 'text' : clip.tokenize([self.text_str_list[idx]], truncate=True)[0]}

    def __len__(self):
        return len(self.text_str_list)
