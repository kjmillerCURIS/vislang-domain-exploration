import os
import sys
import copy
import numpy as np
import pickle
from tqdm import tqdm
from chunk_writer import ChunkWriter
from non_image_data_utils import load_non_image_data
from general_aug_utils import generate_aug_dict
from compute_CLIP_embeddings import BAD_IMAGE_BASES
from do_robustness_and_recovery_analysis import normalize_entire_dict, run_pred, is_it_correct

def do_official_CLIP_zeroshot_analysis(base_dir, embedding_dict_filename_prefix, classifier_npy_filename, stats_dict_filename):

    #easy dicts
    _, class2words_dict, class2filenames_dict = load_non_image_data(base_dir)
    aug_dict = generate_aug_dict()
    classIDs = sorted(class2filenames_dict.keys()) #important that this is SORTED

    #official classifier
    classifier = np.load(classifier_npy_filename)

    #load embeddings
    print('loading image embeddings...')
    my_image_cw = ChunkWriter(None, None, 'image', [], embedding_dict_filename_prefix, readonly=True)
    image_embedding_dict = my_image_cw.load_entire_dict()

    #normalize all embeddings
    print('normalizing image embeddings...')
    image_embedding_dict = normalize_entire_dict(image_embedding_dict)

    aug2acc = {augID : [] for augID in sorted(aug_dict.keys())}
    for augID in tqdm(sorted(aug_dict.keys())):
        for classID in tqdm(sorted(class2filenames_dict.keys())):
            for image_path in tqdm(class2filenames_dict[classID]):
                image_base = os.path.basename(image_path)
                if image_base in BAD_IMAGE_BASES:
                    continue

                image_embedding = image_embedding_dict[(image_base, augID)]
                preds = run_pred(image_embedding, classifier, classIDs)
                aug2acc[augID].append(is_it_correct(preds, classID, 1))

        aug2acc[augID] = 100.0 * np.mean(aug2acc[augID])

    stats_dict = {'primary' : {}, 'secondary' : {}}
    starting_acc = aug2acc['noop']
    stats_dict['secondary']['unaugmented_top1acc_as_percentage'] = starting_acc
    for augID in sorted(aug_dict.keys()):
        if augID == 'noop':
            continue

        stats_dict['primary'][augID] = starting_acc - aug2acc[augID]

    with open(stats_dict_filename, 'wb') as f:
        pickle.dump(stats_dict, f)

def usage():
    print('Usage: python do_official_CLIP_zeroshot_analysis.py <base_dir> <embedding_dict_filename_prefix> <classifier_npy_filename> <stats_dict_filename>')

if __name__ == '__main__':
    do_official_CLIP_zeroshot_analysis(*(sys.argv[1:]))
