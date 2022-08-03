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
    u, s, vh = np.linalg.svd(X - np.mean(X, axis=0)[np.newaxis,:])
    explained_SDs = s / np.sqrt(X.shape[0] - 1)
    components = vh
    return {'explained_SDs' : explained_SDs, 'components' : components, 'spread' : np.linalg.norm(explained_SDs)}

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
def do_direction_decomposition(classIDs, augIDs, pair2vec):
    #idea is that we build up some matrix Q such that Q @ [class_comps ; aug_comps; offset] = [vecs ; 0 ; 0]
    #specifically, Q is grabbing a class component and an aug component for each pair and adding them to the offset
    #the last 2 rows of Q force class_comps and aug_comps to be centered
    #(no, this constraint does not degrade the quality of any solution)

    Q = np.zeros((len(classIDs) * len(augIDs) + 2, len(classIDs) + len(augIDs) + 1))
    
    #centering constraints
    Q[-2,:len(classIDs)] = 1.0 #center classes
    Q[-1,len(classIDs):-1] = 1.0 #center augs
    
    #reconstruction parts
    Q[:-2,-1] = 1.0 #always include offset
    for i in range(len(classIDs)):
        for j in range(len(augIDs)):
            t = i * len(augIDs) + j
            Q[t, i] = 1.0
            Q[t, len(classIDs) + j] = 1.0

    #make target
    embedding_size = len(pair2vec[sorted(pair2vec.keys())[0]]) #yes, this is somewhat sloppy
    Y = np.zeros((len(classIDs) * len(augIDs) + 2, embedding_size))
    for i, classID in enumerate(classIDs):
        for j, augID in enumerate(augIDs):
            t = i * len(augIDs) + j
            Y[t,:] = pair2vec[(classID, augID)]

    #solve
    Xhat = np.dot(np.linalg.pinv(Q), Y)

    #gather learned directions/components
    outputA = {}
    outputA['offset'] = copy.deepcopy(Xhat[-1,:]) #just to be safe
    outputA['class_comps'] = {}
    for i, classID in enumerate(classIDs):
        outputA['class_comps'][classID] = copy.deepcopy(Xhat[i,:])

    for j, augID in enumerate(augIDs):
        outputA['aug_comps'][augID] = copy.deepcopy(Xhat[len(classIDs) + i,:])

    #PCA on residuals
    residuals = (np.dot(Q, Xhat) - Y)[:-2,:]
    outputB = compute_PCA(residuals)

    return outputA, outputB

#returns dict mapping (classID, augID) to either a vector (text) or a list of vectors (image)
def gather_embeddings(class2filenames_dict, class2words_dict, aug_dict, embedding_dict, embedding_type):
    assert(False)

#returns:
#-pair2vec which maps to mean of list of vectors
#-deviation_PCA which is the PCA of the deviations from the means
def reduce_image(pair2vecs):
    assert(False)

#returns:
#-class_PCA
#-aug_PCA
#-total_PCA
def analyze_pair2vec_spreads(pair2vec):
    assert(False)

#you might call this e.g. 4 times:
#-unnormalized, images
#-unnormalized, text
#-normalized, images
#-normalized, text
def do_subspace_analysis_one_embedding_dict(class2filenames_dict, class2words_dict, aug_dict, embedding_dict, embedding_type):
    assert(embedding_type in ['image', 'text'])

    results = {}

    #gather into (classID, augID) pairs
    pair2vec = gather_embeddings(class2filenames_dict, class2words_dict, aug_dict, embedding_dict, embedding_type)
    if embedding_type == 'image':
        pair2vec, results['deviation_PCA'] = reduce_image(pair2vec)

    #compute various spreads with these sets of pairs
    results['class_center_PCA'], results['aug_center_PCA'], results['total_PCA'] = analyze_pair2vec_spreads(pair2vec)

    #decompose into directions, and look at the spread of the residuals
    classIDs = sorted(class2filenames_dict.keys())
    augIDs = sorted(aug_dict.keys())
    direction_decomp, results['direction_residual_PCA'] = do_direction_decomposition(classIDs, augIDs, pair2vec)
    results['direction_decomp'] = direction_decomp

    #now do PCA on the directions and check if they live in orthogonal subspaces
    results['class_comp_PCA'] = compute_PCA(np.array([direction_decomp['class_comps'][classID] for classID in classIDs]))
    results['aug_comp_PCA'] = compute_PCA(np.array([direction_decomp['aug_comps'][augID] for augID in augIDs]))
    results['class_aug_comp_PCA_cossims'] = np.dot(results['class_comp_PCA']['components'], results['aug_comp_PCA']['components'].T)
    
    return results
