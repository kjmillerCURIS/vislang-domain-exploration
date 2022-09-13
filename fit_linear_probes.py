import os
import sys
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['BLIS_NUM_THREADS'] = '2'
import numpy as np
import pickle
import random
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import PredefinedSplit
from chunk_writer import ChunkWriter
from non_image_data_utils import load_non_image_data
from general_aug_utils import generate_aug_dict
from compute_CLIP_embeddings import BAD_IMAGE_BASES, write_to_log_file

DOWNSAMPLE_SEED = 0
DOWNSAMPLE_PROP = 0.25

VAL_SPLIT_SEED = 0
VAL_PROP = 0.1
EMBEDDING_SIZE = 512

DEBUG = False
DEBUG_NUM_IMGS_PER_CLASS = 2

def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)

#X and y should be ready for fitting, i.e. embeddings already normalized, correct dtype, etc.
#returns trained model (as sklearn object)
def fit_probe_to_data(X, y):
    assert(X.dtype == 'float64')
    assert(y.dtype == 'int64')
    assert(len(X.shape) == 2)
    assert(len(y.shape) == 1)
    assert(X.shape[0] == y.shape[0])
    assert(np.amax(np.fabs(np.linalg.norm(X, axis=1) - 1.0)) < 1e-8)
    test_split = -1 * np.ones_like(y)
    random.seed(VAL_SPLIT_SEED)
    for i in np.unique(y):
        indices = np.nonzero(y == i)[0].tolist()
        indices = random.sample(indices, max(int(round(VAL_PROP * len(indices))), 1))
        test_split[indices] = 0

    ps = PredefinedSplit(test_split)
    assert(ps.get_n_splits() == 1)
    write_to_log_file('about_to_fit!\n')
    print('about to fit!')
    my_clf = LogisticRegressionCV(cv=ps, verbose=2)
    my_clf.fit(X, y)
    return my_clf

#returns X, y for one augID
#also returns classIDs, which can be used to turn predictions into classIDs
#the X and y will be suitable for passing into fit_probe_to_data()
def gather_data_one_aug(class2filenames_dict, my_image_cw, augID):
    #first, we gather the image bases and also build y
    y = []
    all_image_bases = []
    classIDs = sorted(class2filenames_dict.keys())
    random.seed(DOWNSAMPLE_SEED)
    for i, classID in enumerate(classIDs):
        image_paths = class2filenames_dict[classID]
        if DEBUG:
            image_paths = image_paths[:DEBUG_NUM_IMGS_PER_CLASS]
        else:
            indices = range(len(image_paths))
            indices = random.sample(indices, max(int(round(DOWNSAMPLE_PROP * len(indices))), 1))
            indices = sorted(indices)
            image_paths = [image_paths[index] for index in indices]

        for image_path in image_paths:
            image_base = os.path.basename(image_path)
            if image_base in BAD_IMAGE_BASES:
                continue

            all_image_bases.append(image_base)
            y.append(i)

    y = np.array(y)
    assert(len(all_image_bases) == len(y))

    #then, we allocate X and populate it with embeddings
    X = np.zeros((len(y), EMBEDDING_SIZE), dtype='float64')
    for t, image_base in tqdm(enumerate(all_image_bases)):
        embedding = my_image_cw.get((image_base, augID))
        embedding = normalize_embedding(embedding)
        X[t,:] = embedding

    return X, y, classIDs

#base_dir should be for IMAGENET TRAINING set
#will do strided iteration through the augmentations, starting at aug_offset index
#for now we'll be safe by giving each aug index its own separate dict file, which will contain a singleton dict
#later some other script can merge them
def fit_linear_probes(base_dir, embedding_dict_filename_prefix, probe_dict_filename_prefix, aug_offset, aug_stride):
    aug_offset = int(aug_offset)
    aug_stride = int(aug_stride)

    _, __, class2filenames_dict = load_non_image_data(base_dir)
    aug_dict = generate_aug_dict()
    augIDs = sorted(aug_dict.keys())

    my_image_cw = ChunkWriter(None, None, 'image', [], embedding_dict_filename_prefix, readonly=True)

    for aug_index in tqdm(range(aug_offset, len(augIDs), aug_stride)):
        augID = augIDs[aug_index]
        probe_dict_filename = probe_dict_filename_prefix + '-%09d.pkl'%(aug_index)
        if os.path.exists(probe_dict_filename):
            print('probe dict "%s" already exists, skipping!'%(probe_dict_filename))
            continue

        X, y, classIDs = gather_data_one_aug(class2filenames_dict, my_image_cw, augID)
        my_clf = fit_probe_to_data(X, y)
        probe_dict = {augID : {'my_clf' : my_clf, 'classIDs' : classIDs}}
        with open(probe_dict_filename, 'wb') as f:
            pickle.dump(probe_dict, f)

def usage():
    print('Usage: python fit_linear_probes.py <base_dir> <embedding_dict_filename_prefix> <probe_dict_filename_prefix> <aug_offset> <aug_stride>')

if __name__ == '__main__':
    fit_linear_probes(*(sys.argv[1:]))
