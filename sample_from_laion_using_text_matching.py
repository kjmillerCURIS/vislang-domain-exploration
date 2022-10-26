import os
import sys
import glob
import heapq
import numpy as np
import pickle
import random
from tqdm import tqdm
from experiment_params.param_utils import get_params_key
from experiment_params.balance_params import grab_params
from compute_CLIP_embeddings import write_to_log_file
from sample_from_laion import populate_url_caption_dicts, save_samples

#returns a list/array with total number to choose from in each domain
def compute_totals(experiment_dir, num_domains):
    totals = np.zeros(num_domains, dtype='int64')
    for shard_filename in tqdm(sorted(glob.glob(os.path.join(experiment_dir, 'domain_text_matching_dict-*.pkl')))):
        with open(shard_filename, 'rb') as f:
            shard = pickle.load(f)

        for k in sorted(shard.keys()):
            totals = totals + shard[k]

    return totals

#return a list of lists of image-bases
#do this by independent sampling, which'll get us a binomially-distributed number of images
#then, do random.sample() to get the exact number desired
#this keeps us from having to have all the image-bases at once in memory
def choose_samples(params, experiment_dir, num_domains):
    p = params
    random.seed(p.laion_sampling_seed)

    #pick probabilities for binomial sampling
    totals = compute_totals(experiment_dir, num_domains)
    with open(os.path.join(experiment_dir, 'text_matching_laion_totals.pkl'), 'wb') as f:
        pickle.dump(totals, f)

    write_to_log_file(str(totals))
    #no need to cap probs at 1, because anything greater than 1 is effectively 1
    probs = p.laion_sampling_safety_factor * p.num_laion_images_per_domain / (1.0 * totals)
    probs[totals <= p.laion_sampling_safety_factor * p.num_laion_images_per_domain] = 1.1

    #do binomial sampling
    samples = [[] for j in range(num_domains)]
    for shard_filename in tqdm(sorted(glob.glob(os.path.join(experiment_dir, 'domain_text_matching_dict-*.pkl')))):
        with open(shard_filename, 'rb') as f:
            shard = pickle.load(f)

        for k in sorted(shard.keys()):
            matches = shard[k]
            for j, sample in enumerate(samples):
                if matches[j] > 0:
                    r = random.uniform(0.0, 1.0)
                    if r < probs[j]:
                        sample.append(k)

    #check that we didn't *needlessly* undersample anything
    assert(all([len(sample) >= p.num_laion_images_per_domain for sample, prob in zip(samples, probs) if prob < 1.0]))

    #sample without replacement to get down to exact number
    final_samples = []
    for sample in samples:
        final_samples.append(random.sample(sample, min(p.num_laion_images_per_domain, len(sample))))

    return final_samples

def sample_from_laion_using_text_matching(experiment_dir, laion_base_dir):
    experiment_dir = os.path.abspath(os.path.expanduser(experiment_dir))
    laion_base_dir = os.path.abspath(os.path.expanduser(laion_base_dir))

    params_key = get_params_key(experiment_dir)
    p = grab_params(params_key)

    assert(p.sampling_method == 'text_matching')

    with open(os.path.join(experiment_dir, 'train_domain_filter.pkl'), 'rb') as f:
        domain_names = pickle.load(f)

    samples = choose_samples(p, experiment_dir, len(domain_names))
    url_caption_dicts = populate_url_caption_dicts(samples, laion_base_dir)
    save_samples(url_caption_dicts, domain_names, experiment_dir)

def usage():
    print('Usage: python sample_from_laion_using_text_matching.py <experiment_dir> <laion_base_dir>')

if __name__ == '__main__':
    sample_from_laion_using_text_matching(*(sys.argv[1:]))
