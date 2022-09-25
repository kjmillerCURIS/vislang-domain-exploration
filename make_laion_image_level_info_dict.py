import os
import sys
import glob
import pandas as pd
import pickle
from tqdm import tqdm

SHARD_SIZE = 5000000

#laion_base_dir should be something like "~/data/vislang-domain-exploration-data/LAION-5B-Subset/ImageEmbeddingsAndMetadata"
#this script will create a pkl file in the same directory, called "image_level_info_dict.pkl"
#key will be image basename
#values will include:
#-embedding_chunk_path (relative to laion_base_dir)
#-metadata_chunk_path (relative to laion_base_dir)
#-laion_type (i.e. what language)
#-file_num (i.e. what index chunk within that language)
#-index_in_chunk
#-caption
#-url
def make_laion_image_level_info_dict(laion_base_dir):
    laion_base_dir = os.path.expanduser(laion_base_dir)

    chunk_level_info_filename = os.path.join(laion_base_dir, 'chunk_level_info.pkl')
    with open(chunk_level_info_filename, 'rb') as f:
        chunk_level_info = pickle.load(f)

    image_level_info_dict = {}
    cur_t = 0
    for laion_type, file_num in tqdm(sorted(chunk_level_info)):
        metadata_chunk_path = os.path.join('Metadata', laion_type, 'metadata_%04d.parquet'%(file_num))
        embedding_chunk_path = os.path.join('ImageEmbeddings', laion_type, 'img_emb_%04d.npy'%(file_num))
        df = pd.read_parquet(os.path.join(laion_base_dir, metadata_chunk_path))
        for index_in_chunk, (caption, url) in enumerate(zip(df['caption'], df['url'])):
            image_level_info = {'embedding_chunk_path' : embedding_chunk_path, 'metadata_chunk_path' : metadata_chunk_path, 'laion_type' : laion_type, 'file_num' : file_num, 'index_in_chunk' : index_in_chunk, 'caption' : caption, 'url' : url}
            image_base = '%s-%04d-%09d.jpg'%(laion_type, file_num, index_in_chunk)
            assert(image_base not in image_level_info_dict)
            image_level_info_dict[image_base] = image_level_info
            if len(image_level_info_dict) >= SHARD_SIZE:
                with open(os.path.join(laion_base_dir, 'image_level_info_dict-%09d.pkl'%(cur_t)), 'wb') as f:
                    pickle.dump(image_level_info_dict, f)

                image_level_info_dict = {}
                cur_t += 1

    image_level_info_dict_filename = os.path.join(laion_base_dir, 'image_level_info_dict.pkl')
    with open(os.path.join(laion_base_dir, 'image_level_info_dict-%09d.pkl'%(cur_t)), 'wb') as f:
        pickle.dump(image_level_info_dict, f)

def usage():
    print('Usage: python make_laion_image_level_info_dict.py <laion_base_dir>')

if __name__ == '__main__':
    make_laion_image_level_info_dict(*(sys.argv[1:]))
