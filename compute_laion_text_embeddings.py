import os
import sys
import clip
import glob
import math
import numpy as np
import pickle
import torch
from tqdm import tqdm
from laion_text_shard_dataset import LaionTextShardDataset
from compute_CLIP_embeddings import write_to_log_file

CLIP_MODEL_TYPE = 'ViT-L/14'
BATCH_SIZE = 256
NUM_WORKERS = 2

def process_one_shard(image_level_info_dict_shard_filename, clip_model):
    out_dirname = os.path.dirname(image_level_info_dict_shard_filename)
    index_part = os.path.splitext(os.path.basename(image_level_info_dict_shard_filename))[0].split('-')[-1]
    out_filename = os.path.join(out_dirname, 'text_embedding_dict-' + index_part + '.pkl')
    if os.path.exists(out_filename):
        print('out shard file "%s" already exists, skipping computation'%(out_filename))
        return

    dataset = LaionTextShardDataset(image_level_info_dict_shard_filename)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
    out_shard = {}
    for batch in tqdm(dataloader):
        text_batch = batch['text'].cuda()
        with torch.no_grad():
            text_embeddings = clip_model.encode_text(text_batch).cpu().numpy()

        image_bases = dataset.get_image_bases(batch['idx'])
        assert(len(image_bases) == len(text_embeddings))
        for image_base, text_embedding in zip(image_bases, text_embeddings):
            out_shard[image_base] = text_embedding

        write_to_log_file('purrpurr! %d'%(len(out_shard)))

    with open(out_filename, 'wb') as f:
        pickle.dump(out_shard, f)

def compute_laion_text_embeddings(laion_base_dir, start_index, stride):
    laion_base_dir = os.path.abspath(os.path.expanduser(laion_base_dir))
    start_index = int(start_index)
    stride = int(stride)

    clip_model, _ = clip.load(CLIP_MODEL_TYPE, device='cuda')
    image_level_info_dict_shard_filenames = sorted(glob.glob(os.path.join(laion_base_dir, 'image_level_info_dict-*.pkl')))
    for image_level_info_dict_shard_filename in tqdm(image_level_info_dict_shard_filenames[start_index::stride]):
        process_one_shard(image_level_info_dict_shard_filename, clip_model)

def usage():
    print('Usage: python compute_laion_text_embeddings.py <laion_base_dir> <start_index> <stride>')

if __name__ == '__main__':
    compute_laion_text_embeddings(*(sys.argv[1:]))
