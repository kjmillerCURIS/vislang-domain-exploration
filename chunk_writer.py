import os
import sys
import numpy as np
import pickle
from tqdm import tqdm

class ChunkWriter:
    #if you're sure that the chunk-index-map file already exists, then you can pass in an empty list for all_keys
    def __init__(self, chunk_size, save_freq, chunk_type, all_keys, filename_prefix, readonly=False):
        assert(chunk_type in ['image', 'text'])
        os.makedirs(os.path.dirname(filename_prefix), exist_ok=True)

        #set params
        self.chunk_size = chunk_size
        self.save_freq = save_freq
        self.chunk_type = chunk_type
        self.filename_prefix = filename_prefix
        self.readonly = readonly

        #make or retrieve chunk_index_map
        chunk_index_map_filename = self.filename_prefix + '-' + self.chunk_type + '-chunk_index_map.pkl'
        if os.path.exists(chunk_index_map_filename):
            with open(chunk_index_map_filename, 'rb') as f:
                self.chunk_index_map = pickle.load(f)

            assert(all([k in self.chunk_index_map for k in all_keys]))
        else:
            if self.readonly:
                print('readonly=True, but expected chunk-index-map file "%s" does not exist!'%(chunk_index_map_filename))
                assert(False)

            self.chunk_index_map = {}
            for t, k in enumerate(all_keys):
                self.chunk_index_map[k] = t // self.chunk_size

            with open(chunk_index_map_filename, 'wb') as f:
                pickle.dump(self.chunk_index_map, f)

        #setup initial state
        self.cur_chunk_index = 0
        cur_chunk_filename = self.filename_prefix + '-' + self.chunk_type + '-chunk_%09d.pkl'%(self.cur_chunk_index)
        if os.path.exists(cur_chunk_filename):
            with open(cur_chunk_filename, 'rb') as f:
                self.cur_chunk = pickle.load(f)

        else:
            self.cur_chunk = {}

        self.save_counter = 0

    def save(self):
        if self.readonly:
            return

        cur_chunk_filename = self.filename_prefix + '-' + self.chunk_type + '-chunk_%09d.pkl'%(self.cur_chunk_index)
        with open(cur_chunk_filename, 'wb') as f:
            pickle.dump(self.cur_chunk, f)

        self.save_counter = 0

    #let index be the chunk-index of key k
    #if cur_chunk_index == index, then do nothing
    #else, save the current chunk, then create or retrieve the one associated with index
    #and set cur_chunk_index to index
    def __update_cur_chunk(self, k):
        index = self.chunk_index_map[k]
        if index == self.cur_chunk_index:
            return

        self.save()
        self.cur_chunk_index = index
        cur_chunk_filename = self.filename_prefix + '-' + self.chunk_type + '-chunk_%09d.pkl'%(self.cur_chunk_index)
        if os.path.exists(cur_chunk_filename):
            with open(cur_chunk_filename, 'rb') as f:
                self.cur_chunk = pickle.load(f)

        else:
            self.cur_chunk = {}

    def contains(self, k):
        self.__update_cur_chunk(k)
        return (k in self.cur_chunk)

    def insert(self, k, v):
        assert(not self.readonly)
        self.__update_cur_chunk(k)
        self.cur_chunk[k] = v
        self.save_counter += 1
        if self.save_counter >= self.save_freq:
            self.save()

    def get(self, k, target_dtype='float64'):
        self.__update_cur_chunk(k)
        v = self.cur_chunk[k]
        if 'torch' in str(v.dtype): #I forgot to do this step for some of the embeddings in the past, so gotta correct for it!
            v = np.squeeze(v.numpy())

        if target_dtype != 'float16': #makes downstream computations more precise. otherwise we'd be using float16 for most/all intermediate steps
            v = v.astype(target_dtype)

        return v

    def load_entire_dict(self, target_dtype='float64'):
        all_keys = sorted(self.chunk_index_map.keys())
        d = {}
        for k in tqdm(all_keys):
            if self.contains(k):
                v = self.get(k, target_dtype=target_dtype)
                d[k] = v

        return d
