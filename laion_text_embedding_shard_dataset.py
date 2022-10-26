import os
import sys
import glob
import numpy as np
import pickle
import torch
from tqdm import tqdm

class LaionTextEmbeddingShardDataset(torch.utils.data.Dataset):

    def __init__(self, text_embedding_shard_filename):
        #load shard
        with open(text_embedding_shard_filename, 'rb') as f:
            shard = pickle.load(f)

        #make parallel lists
        self.image_base_list = []
        self.text_embedding_list = []
        for image_base in sorted(shard.keys()):
            self.image_base_list.append(image_base)
            self.text_embedding_list.append(shard[image_base])

    def __getitem__(self, idx):
        return {'text_embedding' : torch.tensor(self.text_embedding_list[idx], dtype=torch.float32), 'idx' : torch.tensor(idx, dtype=torch.long)}

    def __len__(self):
        return len(self.text_embedding_list)

    #idxs should be a long tensor
    def get_image_bases(self, idxs):
        idxs = idxs.cpu().numpy()
        return [self.image_base_list[idx] for idx in idxs]
