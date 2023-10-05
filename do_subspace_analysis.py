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
from do_robustness_and_recovery_analysis import normalize_entire_dict

#X should be a matrix with each row being a datapoint
#This will return dict with keys:
#-explained_SDs ==> This is the sqrt of what sklearn's "explained_variance_" would be. This is 1/sqrt(N-1) * SVD(centered(X))[1].
#-components ==> The principal components. These are in the rows and should be orthonormal
#-spread ==> np.linalg.norm(explained_SDs)
def compute_PCA(X):
    X = np.array(X)
    X_centered = X - np.mean(X, axis=0)[np.newaxis,:]
    S = np.dot(X_centered.T, X_centered) / (X.shape[0] - 1) #running the SVD on X_centered itself would be too expensive
    u, s, vh = np.linalg.svd(S) #vh will have the eigenvectors in the rows, and s will be the magnitude of the eigenvalues (trust me)
    explained_SDs = np.sqrt(s)
    components = vh
    return {'explained_SDs' : explained_SDs, 'components' : components, 'spread' : np.linalg.norm(explained_SDs)}

#NO LONGER USED
#idea is that we build up some matrix Q such that Q @ [class_comps ; aug_comps; offset] = [vecs ; 0 ; 0]
#specifically, Q is grabbing a class component and an aug component for each pair and adding them to the offset
#the last 2 rows of Q force class_comps and aug_comps to be centered
#(no, this constraint does not degrade the quality of any solution)
def build_Q(num_classes, num_augs):
    assert(False) #NO LONGER USED

    Q = np.zeros((num_classes * num_augs + 2, num_classes + num_augs + 1))
    
    #centering constraints
    Q[-2,:num_classes] = 1.0 #center classes
    Q[-1,num_classes:-1] = 1.0 #center augs
    
    #reconstruction parts
    Q[:-2,-1] = 1.0 #include offet in all reconstruction rows (but not the constraint rows)
    for i in range(num_classes):
        for j in range(num_augs):
            t = i * num_augs + j
            Q[t, i] = 1.0
            Q[t, num_classes + j] = 1.0

    return Q

#pairs should be a list of (classID, augID) pairs
#classID and augID don't necessarily have to be indices. We'll sort the present classes and augs and use those as the indices
#see build_Q() for more details on what this function does
#build_Q_for_pairs() is specifically for cases where pairs might not be a cartesian product
def build_Q_for_pairs(pairs):
    classIDs = sorted(set([p[0] for p in pairs]))
    augIDs = sorted(set([p[1] for p in pairs]))
    num_classes = len(classIDs)
    num_augs = len(augIDs)
    Q_rows = []
    for p in pairs:
        Q_row = np.zeros(num_classes + num_augs + 1)
        Q_row[-1] = 1.0 #this adds in the offset
        Q_row[classIDs.index(p[0])] = 1.0 #this adds in the class component
        Q_row[num_classes + augIDs.index(p[1])] = 1.0 #this adds in the domain component
        Q_rows.append(Q_row)

    #center classes
    Q_row = np.zeros(num_classes + num_augs + 1)
    Q_row[:num_classes] = 1.0
    Q_rows.append(Q_row)

    #center augs
    Q_row = np.zeros(num_classes + num_augs + 1)
    Q_row[num_classes:-1] = 1.0
    Q_rows.append(Q_row)

    return np.array(Q_rows)

#takes in:
#-classIDs as list
#-augIDs as list
#-pair2vec as dictionary mapping (classID, augID) to some vector
#returns 2 dictionaries
#first dictionary with keys:
#-class_comps ==> dictionary mapping classID to component
#-aug_comps ==> dictionary mapping augID to component
#-offset ==> vector which simplifies things by allowing class_comps and aug_comps to be centered
#second dictionary is output of calling compute_PCA() on the residuals
def do_direction_decomposition(pair2vec):
    #(Note: After implementing this, I figured out that this is just the same as averaging out domain/class.
    # But I guess this is more explicit about what it does, and it generalizes better to non-L2 losses.)
    #idea is that we build up some matrix Q such that Q @ [class_comps ; aug_comps; offset] = [vecs ; 0 ; 0]
    #specifically, Q is grabbing a class component and an aug component for each pair and adding them to the offset
    #the last 2 rows of Q force class_comps and aug_comps to be centered
    #(no, this constraint does not degrade the quality of any solution)

    pairs = sorted(pair2vec.keys())
    classIDs = sorted(set([p[0] for p in pairs]))
    augIDs = sorted(set([p[1] for p in pairs]))

    Q = build_Q_for_pairs(pairs)

    #make target
    embedding_size = len(pair2vec[pairs[0]]) #yes, this is somewhat sloppy
    Y = np.zeros((len(pairs) + 2, embedding_size))
    for t, p in enumerate(pairs):
        Y[t,:] = pair2vec[p]

    #solve
    Xhat = np.linalg.pinv(Q) @ Y

    #gather learned directions/components
    outputA = {}
    outputA['offset'] = copy.deepcopy(Xhat[-1,:]) #just to be safe
    outputA['class_comps'] = {}
    outputA['aug_comps'] = {}
    for i, classID in enumerate(classIDs):
        outputA['class_comps'][classID] = copy.deepcopy(Xhat[i,:])

    for j, augID in enumerate(augIDs):
        outputA['aug_comps'][augID] = copy.deepcopy(Xhat[len(classIDs) + j,:])

    #PCA on residuals
    residuals = ((Q @ Xhat) - Y)[:-2,:]
    outputB = compute_PCA(residuals)

    return outputA, outputB

#returns dict mapping (classID, augID) to either a vector (text) or a list of vectors (images)
def gather_embeddings(class2filenames_dict, class2words_dict, aug_dict, embedding_dict, embedding_type, renormalize=False):
    assert(embedding_type in ['image', 'text'])

    #will map (classID, augID) pair to list of vecs
    #if text, then we average these together cuz emsembling is required
    #if images, then we leave the list as-is
    pair2vecs = {}
    for classID in sorted(class2filenames_dict.keys()):
        for augID in sorted(aug_dict.keys()):
            k = (classID, augID)
            if k not in pair2vecs:
                pair2vecs[k] = []

            if embedding_type == 'image':
                for image_path in class2filenames_dict[classID]:
                    image_base = os.path.basename(image_path)
                    if image_base in BAD_IMAGE_BASES:
                        continue

                    kk = (image_base, augID)
                    pair2vecs[k].append(embedding_dict[kk])
            elif embedding_type == 'text':
                for className in class2words_dict[classID]:
                    for text_aug_template in aug_dict[augID]['text_aug_templates']:
                        kk = (classID, className, augID, text_aug_template)
                        pair2vecs[k].append(embedding_dict[kk])
            else:
                assert(False)

    if embedding_type == 'text':
        pair2vec = {}
        for k in sorted(pair2vecs.keys()):
            v = np.mean(pair2vecs[k], axis=0)
            if renormalize:
                v = v / np.linalg.norm(v)

            pair2vec[k] = v

        return pair2vec
    else:
        assert(embedding_type == 'image')
        return pair2vecs

#returns:
#-pair2vec which maps to mean of list of vectors
#-deviation_PCA which is the PCA of the deviations from the means
def reduce_image(pair2vecs):
    pair2vec = {}
    deviations = []
    for k in sorted(pair2vecs.keys()):
        pair2vec[k] = sum(pair2vecs[k]) / len(pair2vecs[k])
        deviations.extend([v - pair2vec[k] for v in pair2vecs[k]])

    deviations = np.array(deviations)
    deviation_PCA = compute_PCA(deviations)
    return pair2vec, deviation_PCA

#returns:
#-class_PCA
#-aug_PCA
#-total_PCA
def analyze_pair2vec_spreads(pair2vec):
    class2vecs = {}
    aug2vecs = {}
    all_vecs = []
    for (classID, augID) in sorted(pair2vec.keys()):
        v = pair2vec[(classID, augID)]
        all_vecs.append(v)
        if classID not in class2vecs:
            class2vecs[classID] = []

        class2vecs[classID].append(v)
        if augID not in aug2vecs:
            aug2vecs[augID] = []

        aug2vecs[augID].append(v)

    X_class = np.array([sum(class2vecs[classID]) / len(class2vecs[classID]) for classID in sorted(class2vecs.keys())])
    X_aug = np.array([sum(aug2vecs[augID]) / len(aug2vecs[augID]) for augID in sorted(aug2vecs.keys())])
    X_total = np.array(all_vecs)
    class_PCA = compute_PCA(X_class)
    aug_PCA = compute_PCA(X_aug)
    total_PCA = compute_PCA(X_total)
    return class_PCA, aug_PCA, total_PCA

#you might call this e.g. 4 times:
#-unnormalized, images
#-unnormalized, text
#-normalized, images
#-normalized, text
def do_subspace_analysis_one_embedding_dict(class2filenames_dict, class2words_dict, aug_dict, embedding_dict, embedding_type, renormalize=False):
    assert(embedding_type in ['image', 'text'])


    #gather into (classID, augID) pairs
    pair2vec = gather_embeddings(class2filenames_dict, class2words_dict, aug_dict, embedding_dict, embedding_type, renormalize=renormalize)

    #do rest of analysis with helper function
    results = do_subspace_analysis_one_embedding_dict_helper(pair2vec, embedding_type, renormalize=renormalize)

    return results

#in case of text, pair2vec should map (classID, augID) to a single vector
#in case of images, pair2vec should map (classID, augID) to a list of vectors
def do_subspace_analysis_one_embedding_dict_helper(pair2vec, embedding_type):
    assert(embedding_type in ['image', 'text'])

    results = {}

    classIDs = sorted(set([p[0] for p in sorted(pair2vec.keys())]))
    augIDs = sorted(set([p[1] for p in sorted(pair2vec.keys())]))

    if embedding_type == 'image':
        pair2vec, results['deviation_PCA'] = reduce_image(pair2vec)

    #compute various spreads with these sets of pairs
    results['class_center_PCA'], results['aug_center_PCA'], results['total_PCA'] = analyze_pair2vec_spreads(pair2vec)

    #decompose into directions, and look at the spread of the residuals
    direction_decomp, results['direction_residual_PCA'] = do_direction_decomposition(pair2vec)
    results['direction_decomp'] = direction_decomp

    #now do PCA on the directions and check if they live in orthogonal subspaces
    results['class_comp_PCA'] = compute_PCA(np.array([direction_decomp['class_comps'][classID] for classID in classIDs]))
    results['aug_comp_PCA'] = compute_PCA(np.array([direction_decomp['aug_comps'][augID] for augID in augIDs]))
    results['class_aug_comp_PCA_cossims'] = np.dot(results['class_comp_PCA']['components'], results['aug_comp_PCA']['components'].T)

    return results

def do_subspace_analysis(base_dir, embedding_dict_filename_prefix, stats_dict_filename):

    #easy dicts
    _, class2words_dict, class2filenames_dict = load_non_image_data(base_dir)
    aug_dict = generate_aug_dict()

    #load embeddings
    print('loading image embeddings...')
    my_image_cw = ChunkWriter(None, None, 'image', [], embedding_dict_filename_prefix, readonly=True)
    image_embedding_dict = my_image_cw.load_entire_dict()
    print('loading text embeddings...')
    my_text_cw = ChunkWriter(None, None, 'text', [], embedding_dict_filename_prefix, readonly=True)
    text_embedding_dict = my_text_cw.load_entire_dict()

    #setup stats_dict
    stats_dict = {'unnormalized' : {}, 'normalized' : {}}

    #do computations (unnormalized)
    stats_dict['unnormalized']['image'] = do_subspace_analysis_one_embedding_dict(class2filenames_dict, class2words_dict, aug_dict, image_embedding_dict, 'image', renormalize=False)
    stats_dict['unnormalized']['text'] = do_subspace_analysis_one_embedding_dict(class2filenames_dict, class2words_dict, aug_dict, text_embedding_dict, 'text', renormalize=False)

    #normalize all embeddings
    print('normalizing image embeddings...')
    image_embedding_dict = normalize_entire_dict(image_embedding_dict)
    print('normalizing text embeddings...')
    text_embedding_dict = normalize_entire_dict(text_embedding_dict)

    #do computations (normalized)
    stats_dict['normalized']['image'] = do_subspace_analysis_one_embedding_dict(class2filenames_dict, class2words_dict, aug_dict, image_embedding_dict, 'image', renormalize=True)
    stats_dict['normalized']['text'] = do_subspace_analysis_one_embedding_dict(class2filenames_dict, class2words_dict, aug_dict, text_embedding_dict, 'text', renormalize=True)

    #save
    with open(stats_dict_filename, 'wb') as f:
        pickle.dump(stats_dict, f)

def usage():
    print('Usage: python do_subspace_analysis.py <base_dir> <embedding_dict_filename_prefix> <stats_dict_filename>')

if __name__ == '__main__':
    do_subspace_analysis(*(sys.argv[1:]))
