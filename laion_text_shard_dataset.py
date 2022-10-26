import os
import sys
import clip
import glob
import numpy as np
import pickle
import random
import torch
from tqdm import tqdm

class LaionTextShardDataset(torch.utils.data.Dataset):

    def __init__(self, image_level_info_dict_shard_filename):
        #load shard
        with open(image_level_info_dict_shard_filename, 'rb') as f:
            shard = pickle.load(f)

        #make parallel lists
        self.image_base_list = []
        self.caption_list = []
        for image_base in sorted(shard.keys()):
            self.image_base_list.append(image_base)
            self.caption_list.append(shard[image_base]['caption'])

    def __getitem__(self, idx):
        caption = self.caption_list[idx]
        tokens = clip.tokenize([caption], truncate=True)[0]
        return {'text' : tokens, 'idx' : torch.tensor(idx, dtype=torch.long)}

    def __len__(self):
        return len(self.caption_list)

    #idxs should be a long tensor
    def get_image_bases(self, idxs):
        idxs = idxs.cpu().numpy()
        return [self.image_base_list[idx] for idx in idxs]
