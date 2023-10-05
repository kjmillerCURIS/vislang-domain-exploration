import os
import sys
import glob
import numpy as np
import pickle
import torch
from tqdm import tqdm
from clip_adapter_utils import make_input
from compute_CLIP_embeddings import write_to_log_file

''' These will be used for evaluation '''

class CorruptedCIFAR10ImageInputOneDomainDataset(torch.utils.data.Dataset):

    #class_domain_dict_filename would be something like CorruptedCIFAR10/test/class_domain_dict.pkl
    #will only emit image inputs of domain non_target
    #setting swap_class_and_domain=True will make it filter by class instead
    def __init__(self, params, image_adapter_input_dict_filename, class_domain_dict_filename, non_target, groups, swap_class_and_domain=False):
        p = params
        self.non_target = non_target
        self.groups = groups
        key_target = 'class'
        key_non_target = 'domain'
        if swap_class_and_domain:
            key_target, key_non_target = key_non_target, key_target

        self.num_layers_to_use = p.num_layers_to_use_for_adapter
        with open(image_adapter_input_dict_filename, 'rb') as f:
            image_adapter_input_dict = pickle.load(f)

        with open(class_domain_dict_filename, 'rb') as f:
            class_domain_dict = pickle.load(f)

        self.image_base_list = []
        self.image_adapter_input_dict = {}
        self.class_list = []
        self.domain_list = []
        for image_base in sorted(image_adapter_input_dict.keys()):
            image_group = (class_domain_dict[image_base][key_target], class_domain_dict[image_base][key_non_target])
            if swap_class_and_domain:
                image_group = (image_group[1], image_group[0])

            if class_domain_dict[image_base][key_non_target] == self.non_target and image_group in self.groups:
                self.image_base_list.append(image_base)
                self.image_adapter_input_dict[image_base] = image_adapter_input_dict[image_base]
                self.class_list.append(class_domain_dict[image_base]['class'])
                self.domain_list.append(class_domain_dict[image_base]['domain'])

    def __getitem__(self, idx):
        image_base = self.image_base_list[idx]
        image_input_obj = self.image_adapter_input_dict[image_base]
        image_input = make_input(image_input_obj, self.num_layers_to_use)
        image_embedding = image_input_obj['embedding']
        return {'image_input' : torch.tensor(image_input, dtype=torch.float32),
                'image_embedding' : torch.tensor(image_embedding, dtype=torch.float32),
                'idx' : torch.tensor(idx, dtype=torch.long)}

    #idxs should be a long tensor
    def get_image_bases(self, idxs):
        idxs = idxs.cpu().numpy()
        return [self.image_base_list[idx] for idx in idxs]

    #idxs should be a long tensor
    def get_classes(self, idxs):
        idxs = idxs.cpu().numpy()
        return [self.class_list[idx] for idx in idxs]

    #idxs should be a long tensor
    def get_domains(self, idxs):
        idxs = idxs.cpu().numpy()
        return [self.domain_list[idx] for idx in idxs]

    def __len__(self):
        return len(self.image_base_list)


class CorruptedCIFAR10TextInputDataset(torch.utils.data.Dataset):

    def __init__(self, params, text_adapter_input_dict_filename, domainful=True):
        p = params
        self.domainful = domainful
        self.num_layers_to_use = p.num_layers_to_use_for_adapter
        with open(text_adapter_input_dict_filename, 'rb') as f:
            self.text_adapter_input_dict = pickle.load(f)

        self.class_list = []
        if self.domainful:
            self.domain_list = []
            for (classID, domain) in sorted(self.text_adapter_input_dict['domainful'].keys()):
                self.class_list.append(classID)
                self.domain_list.append(domain)
        else:
            for classID in sorted(self.text_adapter_input_dict['domainless'].keys()):
                print('HEY YO: append ' + str(classID))
                self.class_list.append(classID)

    def __getitem__(self, idx):
        classID = self.class_list[idx]
        if self.domainful:
            domain = self.domain_list[idx]
            text_input_obj = self.text_adapter_input_dict['domainful'][(classID, domain)]
        else:
            text_input_obj = self.text_adapter_input_dict['domainless'][classID]

        text_input = make_input(text_input_obj, self.num_layers_to_use)
        text_embedding = text_input_obj['embedding']
        return {'text_input' : torch.tensor(text_input, dtype=torch.float32),
                'text_embedding' : torch.tensor(text_embedding, dtype=torch.float32),
                'idx' : torch.tensor(idx, dtype=torch.long)}

    #idxs should be a long tensor
    def get_classes(self, idxs):
        idxs = idxs.cpu().numpy()
        return [self.class_list[idx] for idx in idxs]

    #idxs should be a long tensor
    def get_domains(self, idxs):
        assert(self.domainful)
        idxs = idxs.cpu().numpy()
        return [self.domain_list[idx] for idx in idxs]

    def __len__(self):
        return len(self.class_list)

''' These will be used for subspace analysis '''

class CorruptedCIFAR10ImageInputGroupFilteredDataset(torch.utils.data.Dataset):

    #class_domain_dict_filename would be something like CorruptedCIFAR10/test/class_domain_dict.pkl
    def __init__(self, params, image_adapter_input_dict_filename, class_domain_dict_filename, groups):
        p = params
        self.groups = groups
        self.num_layers_to_use = p.num_layers_to_use_for_adapter
        with open(image_adapter_input_dict_filename, 'rb') as f:
            image_adapter_input_dict = pickle.load(f)

        with open(class_domain_dict_filename, 'rb') as f:
            class_domain_dict = pickle.load(f)

        self.image_base_list = []
        self.image_adapter_input_dict = {}
        self.class_list = []
        self.domain_list = []
        for image_base in sorted(image_adapter_input_dict.keys()):
            image_group = (class_domain_dict[image_base]['class'], class_domain_dict[image_base]['domain'])
            if image_group in self.groups:
                self.image_base_list.append(image_base)
                self.image_adapter_input_dict[image_base] = image_adapter_input_dict[image_base]
                self.class_list.append(class_domain_dict[image_base]['class'])
                self.domain_list.append(class_domain_dict[image_base]['domain'])

    def __getitem__(self, idx):
        image_base = self.image_base_list[idx]
        image_input_obj = self.image_adapter_input_dict[image_base]
        image_input = make_input(image_input_obj, self.num_layers_to_use)
        image_embedding = image_input_obj['embedding']
        return {'input' : torch.tensor(image_input, dtype=torch.float32),
                'embedding' : torch.tensor(image_embedding, dtype=torch.float32),
                'idx' : torch.tensor(idx, dtype=torch.long)}

    #idxs should be a long tensor
    def get_image_bases(self, idxs):
        idxs = idxs.cpu().numpy()
        return [self.image_base_list[idx] for idx in idxs]

    #idxs should be a long tensor
    def get_classes(self, idxs):
        idxs = idxs.cpu().numpy()
        return [self.class_list[idx] for idx in idxs]

    #idxs should be a long tensor
    def get_domains(self, idxs):
        idxs = idxs.cpu().numpy()
        return [self.domain_list[idx] for idx in idxs]

    def __len__(self):
        return len(self.image_base_list)

class CorruptedCIFAR10TextInputGroupFilteredDataset(torch.utils.data.Dataset):

    def __init__(self, params, text_adapter_input_dict_filename, groups):
        p = params
        self.num_layers_to_use = p.num_layers_to_use_for_adapter
        with open(text_adapter_input_dict_filename, 'rb') as f:
            self.text_adapter_input_dict = pickle.load(f)

        self.class_list = []
        self.domain_list = []
        for (classID, domain) in sorted(self.text_adapter_input_dict['domainful'].keys()):
            if (classID, domain) in groups:
                self.class_list.append(classID)
                self.domain_list.append(domain)

    def __getitem__(self, idx):
        classID = self.class_list[idx]
        domain = self.domain_list[idx]
        text_input_obj = self.text_adapter_input_dict['domainful'][(classID, domain)]
        text_input = make_input(text_input_obj, self.num_layers_to_use)
        text_embedding = text_input_obj['embedding']
        return {'input' : torch.tensor(text_input, dtype=torch.float32),
                'embedding' : torch.tensor(text_embedding, dtype=torch.float32),
                'idx' : torch.tensor(idx, dtype=torch.long)}

    #idxs should be a long tensor
    def get_classes(self, idxs):
        idxs = idxs.cpu().numpy()
        return [self.class_list[idx] for idx in idxs]

    #idxs should be a long tensor
    def get_domains(self, idxs):
        idxs = idxs.cpu().numpy()
        return [self.domain_list[idx] for idx in idxs]

    def __len__(self):
        return len(self.class_list)
