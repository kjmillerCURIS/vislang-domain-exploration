import os
import sys
import glob
import numpy as np
import pickle
import torch
from tqdm import tqdm
from clip_adapter_utils import make_input
from compute_CLIP_embeddings import write_to_log_file

class CorruptedCIFAR10DisentanglementTextDataset(torch.utils.data.Dataset):

    def __init__(self, params, text_adapter_input_dict_filename, splits_filename, split_type='trivial', split_index=None):
        p = params
        self.num_layers_to_use = p.num_layers_to_use_for_adapter
        with open(text_adapter_input_dict_filename, 'rb') as f:
            self.text_adapter_input_dict = pickle.load(f)

        with open(splits_filename, 'rb') as f:
            splits = pickle.load(f)

        if split_type == 'trivial':
            self.train_groups = splits[split_type]['train']
        else:
            assert(split_type in ['easy_zeroshot', 'hard_zeroshot'])
            self.train_groups = splits[split_type][split_index]['train']

        #you can use these from outside to figure out num_classes and num_domains
        self.classes = sorted(set([g[0] for g in self.train_groups]))
        self.domains = sorted(set([g[1] for g in self.train_groups]))

    def __getitem__(self, idx):
        (classID, domain) = self.train_groups[idx]
        class_index = self.classes.index(classID)
        domain_index = self.domains.index(domain)
        text_input_obj = self.text_adapter_input_dict['domainful'][(classID, domain)]
        text_input = make_input(text_input_obj, self.num_layers_to_use)
        text_embedding = text_input_obj['embedding']
        return {'input' : torch.tensor(text_input, dtype=torch.float32),
                'embedding' : torch.tensor(text_embedding, dtype=torch.float32),
                'class' : torch.tensor(class_index, dtype=torch.long),
                'domain' : torch.tensor(domain_index, dtype=torch.long)}

    def __len__(self):
        return len(self.train_groups)
