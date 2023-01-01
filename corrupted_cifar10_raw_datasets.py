import os
import sys
import glob
import numpy as np
import pickle
from PIL import Image
import torch
from tqdm import tqdm
import clip
from non_image_data_utils_corrupted_cifar10 import CLASS_NAMES
from general_aug_utils_corrupted_cifar10 import generate_aug_dict, generate_domainless_text_template
from compute_CLIP_embeddings import write_to_log_file

#'idx' : torch.tensor(idx, dtype=torch.long)

CLIP_MODEL_TYPE = 'ViT-B/32'

class CorruptedCIFAR10RawImageDataset(torch.utils.data.Dataset):
    
    #src_dir should have basename like "train" or "test"
    def __init__(self, src_dir):
        _, self.preprocess = clip.load(CLIP_MODEL_TYPE, device='cpu')
        self.image_dir = os.path.join(src_dir, 'images')
        self.image_base_list = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(self.image_dir, '*.png')))]

    def __getitem__(self, idx):
        image_filename = os.path.join(self.image_dir, self.image_base_list[idx])
        return {'image' : self.preprocess(Image.open(image_filename)).type(torch.float16), 'idx' : torch.tensor(idx, dtype=torch.long)}

    def __len__(self):
        return len(self.image_base_list)

    #idxs should be a long tensor
    def get_image_bases(self, idxs):
        idxs = idxs.cpu().numpy()
        return [self.image_base_list[idx] for idx in idxs]

class CorruptedCIFAR10RawTextDataset(torch.utils.data.Dataset):

    def __init__(self, domainful=True):
        self.domainful = domainful
        self.class_names = CLASS_NAMES
        if self.domainful:
            self.aug_dict = generate_aug_dict()
            self.class_list = []
            self.domain_list = []
            for classID in sorted(self.class_names.keys()):
                for domain in sorted(self.aug_dict.keys()):
                    self.class_list.append(classID)
                    self.domain_list.append(domain)
        else:
            self.domainless_text_template = generate_domainless_text_template()
            self.class_list = sorted(self.class_names.keys())

    def __getitem__(self, idx):
        classID = self.class_list[idx]
        if self.domainful:
            domain = self.domain_list[idx]
            text = self.aug_dict[domain]['text_aug_template'] % self.class_names[classID]
        else:
            text = self.domainless_text_template % self.class_names[classID]

        return {'text' : clip.tokenize([text], truncate=True)[0], 'idx' : torch.tensor(idx, dtype=torch.long)}

    def __len__(self):
        return len(self.class_list) #works for both self.domainful=True and self.domainful=False

    #idxs should be a long tensor
    def get_classes(self, idxs):
        idxs = idxs.cpu().numpy()
        return [self.class_list[idx] for idx in idxs]

    #idxs should be a long tensor
    def get_domains(self, idxs):
        assert(self.domainful)
        idxs = idxs.cpu().numpy()
        return [self.domain_list[idx] for idx in idxs]
