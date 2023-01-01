import os
import sys
import glob
import numpy as np
import pickle
import random
import torch
from tqdm import tqdm
from clip_adapter_utils import make_input
from compute_CLIP_embeddings import write_to_log_file

DOMAINLESS_TEXT_SEED = 0

class CorruptedCIFAR10InputPairDataset(torch.utils.data.Dataset):

    #class_domain_dict_filename would be something like CorruptedCIFAR10/train/class_domain_dict.pkl
    def __init__(self, params, image_adapter_input_dict_filename, text_adapter_input_dict_filename, class_domain_dict_filename):
        p = params
        self.num_layers_to_use = p.num_layers_to_use_for_adapter
        with open(image_adapter_input_dict_filename, 'rb') as f:
            self.image_adapter_input_dict = pickle.load(f)

        with open(text_adapter_input_dict_filename, 'rb') as f:
            self.text_adapter_input_dict = pickle.load(f)

        with open(class_domain_dict_filename, 'rb') as f:
            self.class_domain_dict = pickle.load(f)

        self.image_base_list = sorted(self.image_adapter_input_dict.keys())
        self.__setup_domainless_text_set(p)

    def __setup_domainless_text_set(self, params):
        p = params
        self.domainless_text_set = set([])
        if p.domainless_text_prop == 0.0:
            return

        random.seed(DOMAINLESS_TEXT_SEED)
        buckets = {}
        for image_base in self.image_base_list:
            classID = self.class_domain_dict[image_base]['class']
            domain = self.class_domain_dict[image_base]['domain']
            k = (classID, domain)
            if k not in buckets:
                buckets[k] = []

            buckets[k].append(image_base)

        for k in sorted(buckets.keys()):
            image_bases_to_add = random.sample(buckets[k], int(round(p.domainless_text_prop * len(buckets[k]))))
            for image_base in image_bases_to_add:
                self.domainless_text_set.add(image_base)

    def __getitem__(self, idx):
        image_base = self.image_base_list[idx]
        image_input_obj = self.image_adapter_input_dict[image_base]
        image_input = make_input(image_input_obj, self.num_layers_to_use)
        image_embedding = image_input_obj['embedding']
        classID = self.class_domain_dict[image_base]['class']
        domain = self.class_domain_dict[image_base]['domain']
        if image_base in self.domainless_text_set:
            text_input_obj = self.text_adapter_input_dict['domainless'][classID]
        else:
            text_input_obj = self.text_adapter_input_dict['domainful'][(classID, domain)]

        text_input = make_input(text_input_obj, self.num_layers_to_use)
        text_embedding = text_input_obj['embedding']
        return {'image_input' : torch.tensor(image_input, dtype=torch.float32),
                'image_embedding' : torch.tensor(image_embedding, dtype=torch.float32),
                'text_input' : torch.tensor(text_input, dtype=torch.float32),
                'text_embedding' : torch.tensor(text_embedding, dtype=torch.float32)}

    def __len__(self):
        return len(self.image_base_list)
