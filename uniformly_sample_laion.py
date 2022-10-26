import os
import sys
import glob
import pickle
import random
from tqdm import tqdm
from experiment_params.balance_params import grab_params
from experiment_params.param_utils import get_params_key
from sample_from_laion import populate_url_caption_dicts, save_samples
from compute_CLIP_embeddings import write_to_log_file

#populate_url_caption_dicts(samples, laion_base_dir):
#save_samples(url_caption_dicts, domain_names, experiment_dir, include_domain_index=True):

def choose_sample(params, laion_base_dir, num_images_to_sample, random_seed):
    p = params
    dict_filenames = sorted(glob.glob(os.path.join(laion_base_dir, 'image_level_info_dict-*.pkl')))
    image_bases = []
    for dict_filename in tqdm(dict_filenames):
        with open(dict_filename, 'rb') as f:
            image_level_info_dict = pickle.load(f)

        if p.english_only:
            image_bases.extend([k for k in sorted(image_level_info_dict.keys()) if image_level_info_dict[k]['laion_type'] == 'laion2B-en'])
        else:
            image_bases.extend(sorted(image_level_info_dict.keys()))

    random.seed(random_seed)
    sample = random.sample(image_bases, num_images_to_sample)
    return sample

def uniformly_sample_laion(experiment_dir, laion_base_dir):
    experiment_dir = os.path.abspath(os.path.expanduser(experiment_dir))
    laion_base_dir = os.path.abspath(os.path.expanduser(laion_base_dir))

    p = grab_params(get_params_key(experiment_dir))

    assert(p.sampling_method == 'uniform')

    sample = choose_sample(p, laion_base_dir, p.uniform_sample_size, p.laion_sampling_seed)
    url_caption_dicts = populate_url_caption_dicts([sample], laion_base_dir)
    save_samples(url_caption_dicts, ['uniform_subset'], experiment_dir, include_domain_index=False)

def usage():
    print('Usage: python uniformly_sample_laion.py <experiment_dir> <laion_base_dir>')

if __name__ == '__main__':
    uniformly_sample_laion(*(sys.argv[1:]))
