import os
import sys
import clip
import torch
from tqdm import tqdm
from non_image_data_utils import load_non_image_data
from general_aug_utils import generate_aug_dict

class DisentanglementTextDataset(torch.utils.data.Dataset):
    ''' give X, class, domain, where X is textual input, and class and domain are indices '''

    #we have image_base_dir just so we can get the class names, even though we're not dealing with images
    def __init__(self, image_base_dir, class_filter=None, domain_filter=None):
        self.raw_texts = []
        self.class_indices = []
        self.domain_indices = []

        #get necessary dictionaries for forming texts
        _, class2words_dict, __ = load_non_image_data(image_base_dir)
        aug_dict = generate_aug_dict()

        #setup filters
        if class_filter is None:
            class_filter = sorted(class2words_dict.keys())

        if domain_filter is None:
            domain_filter = sorted(aug_dict.keys())

        #make texts!
        for class_index, classID in tqdm(enumerate(class_filter)):
            for className in class2words_dict[classID]:
                for domain_index, augID in enumerate(domain_filter):
                    for text_aug_template in aug_dict[augID]['text_aug_templates']:
                        self.raw_texts.append(text_aug_template % className)
                        self.class_indices.append(class_index)
                        self.domain_indices.append(domain_index)

    def __getitem__(self, idx):
        sample = {}
        sample['X'] = clip.tokenize([self.raw_texts[idx]], truncate=True)[0]
        sample['class'] = torch.tensor(self.class_indices[idx], dtype=torch.long)
        sample['domain'] = torch.tensor(self.domain_indices[idx], dtype=torch.long)
        return sample

    def __len__(self):
        return len(self.raw_texts)
