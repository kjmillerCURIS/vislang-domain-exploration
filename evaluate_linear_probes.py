import os
import sys
import copy
import glob
import numpy as np
import pickle
from tqdm import tqdm
from chunk_writer import ChunkWriter
from non_image_data_utils import load_non_image_data
from general_aug_utils import generate_aug_dict
from compute_CLIP_embeddings import BAD_IMAGE_BASES
from do_robustness_and_recovery_analysis import normalize_entire_dict, run_pred, is_it_correct

def load_probe_dict(probe_dict_filename_prefix):
    probe_dict = {}
    filenames = sorted(glob.glob(probe_dict_filename_prefix + '-*.pkl'))
    for filename in tqdm(filenames):
        with open(filename, 'rb') as f:
            probe_dict_one = pickle.load(f)

        for k in sorted(probe_dict_one.keys()):
            if k in probe_dict:
                print('!')

            probe_dict[k] = probe_dict_one[k]

    return probe_dict

#base_dir should be for IMAGENET VALIDATION
#same for embedding_dict_filename_prefix
#will do a glob *.pkl on probe_dict_filename_prefix to find all the shards and merge into one probe dict
def evaluate_linear_probes(base_dir, embedding_dict_filename_prefix, probe_dict_filename_prefix, stats_dict_filename):

    #easy dicts
    _, class2words_dict, class2filenames_dict = load_non_image_data(base_dir)
    aug_dict = generate_aug_dict()

    #probes
    probe_dict = load_probe_dict(probe_dict_filename_prefix)

    #load embeddings
    print('loading image embeddings...')
    my_image_cw = ChunkWriter(None, None, 'image', [], embedding_dict_filename_prefix, readonly=True)
    image_embedding_dict = my_image_cw.load_entire_dict()

    #normalize all embeddings
    print('normalizing image embeddings...')
    image_embedding_dict = normalize_entire_dict(image_embedding_dict)

    stats_dict = {augID : [] for augID in sorted(aug_dict.keys())}
    for augID in tqdm(sorted(aug_dict.keys())):
        for classID in tqdm(sorted(class2filenames_dict.keys())):
            for image_path in tqdm(class2filenames_dict[classID]):
                image_base = os.path.basename(image_path)
                if image_base in BAD_IMAGE_BASES:
                    continue

                image_embedding = image_embedding_dict[(image_base, augID)]
                preds = run_pred(image_embedding, probe_dict[augID]['my_clf'], probe_dict[augID]['classIDs'])
                stats_dict[augID].append(is_it_correct(preds, classID, 1))

        stats_dict[augID] = 100.0 * np.mean(stats_dict[augID])

    with open(stats_dict_filename, 'wb') as f:
        pickle.dump(stats_dict, f)

def usage():
    print('Usage: python evaluate_linear_probes.py <base_dir> <embedding_dict_filename_prefix> <probe_dict_filename_prefix> <stats_dict_filename>')

if __name__ == '__main__':
    evaluate_linear_probes(*(sys.argv[1:]))
