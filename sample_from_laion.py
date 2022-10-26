import os
import sys
import glob
import heapq
import numpy as np
import pickle
from tqdm import tqdm
from experiment_params.param_utils import get_params_key
from experiment_params.balance_params import grab_params
from compute_CLIP_embeddings import write_to_log_file

'''
This will pick the top N laion images for each domain based on their domain log prob.
This will create a directory structure <experiment_dir>/laion_sample/[<domain_num>-<domain_name>]
Each domain directory will have the following files:
- "image_urls.txt" which is a plaintext list of image urls
- "image_bases.pkl" which is a pickle of a list of image bases (same order as image_urls.txt)
- "image_base_to_caption.pkl" which is a pickle of a dict mapping image bases to captions
'''

#return a list of lists of image-bases
#do this by loading the log_prob_dict shards, one at a time, and populating priority-queues
def choose_samples(experiment_dir, num_domains, num_images_per_domain):
    #the heaps (i.e. priority queues)
    h_list = []
    for j in range(num_domains):
        h_list.append([])

    #go through each shard
    log_probs_dict_filenames = sorted(glob.glob(os.path.join(experiment_dir, 'laion_log_probs_dict-*.pkl')))
    write_to_log_file('found %d shards'%(len(log_probs_dict_filenames)))
    for log_probs_dict_filename in log_probs_dict_filenames:
        write_to_log_file('loading from "%s"...'%(log_probs_dict_filename))
        with open(log_probs_dict_filename, 'rb') as f:
            sub_dict = pickle.load(f)

        write_to_log_file('done loading from "%s"'%(log_probs_dict_filename))
        write_to_log_file('populating priority queue...')
        for image_base in sorted(sub_dict.keys()):
            log_probs = sub_dict[image_base]
            for log_prob, h in zip(log_probs, h_list):
                if len(h) < num_images_per_domain:
                    heapq.heappush(h, (log_prob, image_base))
                else:
                    heapq.heappushpop(h, (log_prob, image_base))

        write_to_log_file('done populating priority queue')

    print('lotsa popping and returning...')
    return [[heapq.heappop(h)[1] for i in range(len(h))][::-1] for h in h_list]

#samples should be a list of lists of image-bases
#this will return a list of dicts (one per domain) mapping each image_base to a (url, caption) tuple
def populate_url_caption_dicts(samples, laion_base_dir):
    write_to_log_file('populating url-caption dicts...')
    url_caption_dicts = [{} for k in range(len(samples))]
    for dict_filename in sorted(glob.glob(os.path.join(laion_base_dir, 'image_level_info_dict-*.pkl'))):
        write_to_log_file('loading dict "%s"...'%(dict_filename))
        with open(dict_filename, 'rb') as f:
            image_level_info_dict = pickle.load(f)

        write_to_log_file('done loading dict "%s"'%(dict_filename))
        write_to_log_file('populating...')
        for sample, url_caption_dict in zip(samples, url_caption_dicts):
            for image_base in sample:
                if image_base in image_level_info_dict:
                    assert(image_base not in url_caption_dict)
                    url_caption_dict[image_base] = (image_level_info_dict[image_base]['url'], image_level_info_dict[image_base]['caption'])

        write_to_log_file('done populating')

    write_to_log_file('done populating url-caption dicts')
    return url_caption_dicts

def save_samples(url_caption_dicts, domain_names, experiment_dir, include_domain_index=True):
    sample_base_dir = os.path.join(experiment_dir, 'laion_sample')
    os.makedirs(sample_base_dir, exist_ok=True)
    for i, (url_caption_dict, domain_name) in tqdm(enumerate(zip(url_caption_dicts, domain_names))):
        if include_domain_index:
            sample_dir = os.path.join(sample_base_dir, '%03d-%s'%(i, domain_name))
        else:
            sample_dir = os.path.join(sample_base_dir, domain_name)

        os.makedirs(sample_dir, exist_ok=True)
        write_to_log_file('saving for domian %d...'%(i))
        image_bases = sorted(url_caption_dict.keys())
        with open(os.path.join(sample_dir, 'image_base_to_caption.pkl'), 'wb') as f:
            pickle.dump({image_base : url_caption_dict[image_base][1] for image_base in image_bases}, f)

        with open(os.path.join(sample_dir, 'image_bases.pkl'), 'wb') as f:
            pickle.dump(image_bases, f)

        with open(os.path.join(sample_dir, 'image_urls.txt'), 'w') as f:
            f.write('\n'.join([url_caption_dict[image_base][0].strip() for image_base in image_bases]))

        write_to_log_file('done saving for domain %d'%(i))

#returns log_probs_dict, max_shard_index
#this is used by at least one other script (compute_domain_log_probs_for_laion.py), even if it's not used by this script anymore
def load_log_probs_dict(experiment_dir):
    log_probs_dict_filenames = sorted(glob.glob(os.path.join(experiment_dir, 'laion_log_probs_dict-*.pkl')))
    log_probs_dict = {}
    max_shard_index = -1
    for log_probs_dict_filename in log_probs_dict_filenames:
        write_to_log_file('loading from "%s"...'%(log_probs_dict_filename))
        shard_index = int(os.path.splitext(os.path.basename(log_probs_dict_filename))[0].split('-')[-1])
        max_shard_index = max(shard_index, max_shard_index)
        with open(log_probs_dict_filename, 'rb') as f:
            sub_dict = pickle.load(f)

        write_to_log_file('done loading from "%s"'%(log_probs_dict_filename))
        for image_base in sorted(sub_dict.keys()):
            log_probs_dict[image_base] = sub_dict[image_base]

    return log_probs_dict, max_shard_index

def sample_from_laion(experiment_dir, laion_base_dir):
    experiment_dir = os.path.expanduser(experiment_dir)
    laion_base_dir = os.path.expanduser(laion_base_dir)

    params_key = get_params_key(experiment_dir)
    p = grab_params(params_key)
    assert(p.sampling_method == 'classifier')

    with open(os.path.join(experiment_dir, 'train_domain_filter.pkl'), 'rb') as f:
        domain_names = pickle.load(f)

    samples = choose_samples(experiment_dir, len(domain_names), p.num_laion_images_per_domain)
    url_caption_dicts = populate_url_caption_dicts(samples, laion_base_dir)
    save_samples(url_caption_dicts, domain_names, experiment_dir)

def usage():
    print('Usage: python sample_from_laion.py <experiment_dir> <laion_base_dir>')

if __name__ == '__main__':
    sample_from_laion(*(sys.argv[1:]))
