import os
import sys
import random
import requests
import pickle
from tqdm import tqdm

TOTAL_NUM_FILES_DICT = {'laion2B-en' : 2314, 'laion2B-multi' : 2267, 'laion1B-nolang' : 1273}
NUM_FILES_TO_DOWNLOAD_DICT = {'laion2B-en' : 100, 'laion2B-multi' : 100, 'laion1B-nolang' : 50}
RANDOM_SEED_DICT = {'laion2B-en' : 0, 'laion2B-multi' : 42, 'laion1B-nolang' : 1337}

def sample_pairs():
    my_pairs = []
    for laion_type in sorted(TOTAL_NUM_FILES_DICT.keys()):
        total_num_files = TOTAL_NUM_FILES_DICT[laion_type]
        num_files_to_download = NUM_FILES_TO_DOWNLOAD_DICT[laion_type]
        random_seed = RANDOM_SEED_DICT[laion_type]
        random.seed(random_seed)
        file_nums = random.sample(range(total_num_files), num_files_to_download)
        my_pairs.extend([(laion_type, file_num) for file_num in file_nums])

    return my_pairs

#laion_type should be 'laion2B-en', 'laion2B-multi', or 'laion1B-nolang'
#file_num should be some int that goes into the filename as 4 digits
#returns img_emb_url, metadata_url
def get_urls(laion_type, file_num):
    prefix = 'https://mystic.the-eye.eu/public/AI/cah/laion5b/embeddings/'
    img_emb_url = prefix + laion_type + '/img_emb/img_emb_%04d.npy'%(file_num)
    metadata_url = prefix + laion_type + '/' + laion_type + '-metadata/metadata_%04d.parquet'%(file_num)
    return img_emb_url, metadata_url

def download_one(my_url, dst_filename):
    my_response = requests.get(my_url)
    with open(dst_filename, 'wb') as f:
        f.write(my_response.content)

def download_pair(my_pair, img_emb_dir, metadata_dir):
    laion_type, file_num = my_pair
    img_emb_subdir = os.path.join(img_emb_dir, laion_type)
    os.makedirs(img_emb_subdir, exist_ok=True)
    metadata_subdir = os.path.join(metadata_dir, laion_type)
    os.makedirs(metadata_subdir, exist_ok=True)
    img_emb_url, metadata_url = get_urls(laion_type, file_num)
    download_one(img_emb_url, os.path.join(img_emb_subdir, 'img_emb_%04d.npy'%(file_num)))
    download_one(metadata_url, os.path.join(metadata_subdir, 'metadata_%04d.parquet'%(file_num)))

def download_laion5b_subset_embeddings(dst_dir):
    img_emb_dir = os.path.join(dst_dir, 'ImageEmbeddings')
    metadata_dir = os.path.join(dst_dir, 'Metadata')
    os.makedirs(img_emb_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    info_filename = os.path.join(dst_dir, 'chunk_level_info.pkl') #this will have a set of (laion_type, file_num) pairs that have been downloaded
    if os.path.exists(info_filename):
        with open(info_filename, 'rb') as f:
            info = pickle.load(f)
    else:
        info = set([])

    my_pairs = sample_pairs()
    my_pairs = [p for p in my_pairs if p not in info]
    for my_pair in tqdm(my_pairs):
        download_pair(my_pair, img_emb_dir, metadata_dir)
        info.add(my_pair)
        with open(info_filename, 'wb') as f:
            pickle.dump(info, f)

def usage():
    print('Usage: python download_laion5b_subset_embeddings.py <dst_dir>')

if __name__ == '__main__':
    download_laion5b_subset_embeddings(*(sys.argv[1:]))
