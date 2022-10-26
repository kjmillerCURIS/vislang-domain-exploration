import os
import sys
import glob
import numpy as np
import pickle
import random
import torch
from tqdm import tqdm

#shuffling option in case we only have time to process a subset of the data, at least it'll be a roughly uniform subset
SHUFFLE_IMAGE_LEVEL_INFO_DICTS = True

class LaionImageEmbeddingDataset(torch.utils.data.Dataset):

    #make parallel lists
    def __make_lists(self, laion_base_dir, already_seen=set([])):
        self.embedding_chunk_path_list = []
        self.index_in_chunk_list = []
        self.image_base_list = []
        print('loading a biiiiiig pickle (well, actually, lots of moderate pickles)...')
        dict_filenames = sorted(glob.glob(os.path.join(laion_base_dir, 'image_level_info_dict-*.pkl')))
        if SHUFFLE_IMAGE_LEVEL_INFO_DICTS:
            random.shuffle(dict_filenames)

        for dict_filename in dict_filenames:
            with open(dict_filename, 'rb') as f:
                image_level_info_dict = pickle.load(f)

            for image_base in sorted(image_level_info_dict.keys()):
                if image_base in already_seen:
                    continue

                self.image_base_list.append(image_base)
                self.embedding_chunk_path_list.append(image_level_info_dict[image_base]['embedding_chunk_path'])
                self.index_in_chunk_list.append(image_level_info_dict[image_base]['index_in_chunk'])

        print('well that was huge (and so was that)')

    def __init__(self, laion_base_dir, already_seen=set([])):
        self.__make_lists(laion_base_dir, already_seen=already_seen)
        self.laion_base_dir = laion_base_dir
        self.cur_embedding_chunk_path = 'MEOWMEOWMEOW'
        self.cur_embedding_chunk = None

    def __getitem__(self, idx):
        embedding_chunk_path = self.embedding_chunk_path_list[idx]
        if embedding_chunk_path != self.cur_embedding_chunk_path:
            self.cur_embedding_chunk_path = embedding_chunk_path
            self.cur_embedding_chunk = np.load(os.path.join(self.laion_base_dir, embedding_chunk_path))

        embedding = self.cur_embedding_chunk[self.index_in_chunk_list[idx]]
        return {'embedding' : torch.tensor(embedding, dtype=torch.float32), 'idx' : torch.tensor(idx, dtype=torch.long)}

    def __len__(self):
        return len(self.image_base_list)

    #idxs should be a long tensor
    def get_image_bases(self, idxs):
        idxs = idxs.cpu().numpy()
        return [self.image_base_list[idx] for idx in idxs]
