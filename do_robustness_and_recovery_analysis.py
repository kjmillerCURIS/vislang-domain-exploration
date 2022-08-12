import os
import sys
import numpy as np
import pickle
from scipy.stats import ttest_rel
from tqdm import tqdm
from chunk_writer import ChunkWriter
from non_image_data_utils import load_non_image_data
from general_aug_utils import generate_aug_dict
from compute_CLIP_embeddings import BAD_IMAGE_BASES

MAX_TOP_K = 10

#CAUTION: This modifies embedding_dict IN-PLACE!!!!!!!
def normalize_entire_dict(embedding_dict):
    for k in tqdm(sorted(embedding_dict.keys())):
        embedding_dict[k] = embedding_dict[k] / np.linalg.norm(embedding_dict[k])

    return embedding_dict

#returns top-MAX_TOP_K-scoring classIDs, in order
def run_pred(image_embedding, classifier, classIDs):
    cossims = np.squeeze(np.dot(classifier, image_embedding[:,np.newaxis]))
    top_indices = np.argsort(-cossims)[:MAX_TOP_K]
    return [classIDs[i] for i in top_indices]

#returns the following:
#-A_dict: maps image_base to pred_class, given unaugmented image with standard prompt
#-B_dict: maps augID, then image_base to pred_class, given augmented image with standard prompt
#-C_dict: maps augID, then image_base to pred_class, given augmented image with augmented prompt
def compute_zeroshot_preds(class2filenames_dict, class2words_dict, aug_dict, image_embedding_dict, text_embedding_dict):
    embedding_size = len(text_embedding_dict[sorted(text_embedding_dict.keys())[0]])
    
    #setup classifiers as matrices
    #this will be a dictionary mapping each augID to a matrix
    #will initially be classifiers[augID][classID] = list
    #we'll average and shape into matrices as we go
    classIDs = sorted(class2filenames_dict.keys())
    classifiers = {}
    for augID in sorted(aug_dict.keys()):
        classifiers[augID] = {}
        for classID in classIDs:
            classifiers[augID][classID] = []
            for className in class2words_dict[classID]:
                for text_aug_template in aug_dict[augID]['text_aug_templates']:
                    k = (classID, className, augID, text_aug_template)
                    classifiers[augID][classID].append(text_embedding_dict[k])

            #average and renormalize
            v = np.mean(classifiers[augID][classID], axis=0)
            v = v / np.linalg.norm(v)
            classifiers[augID][classID] = v

        W = np.zeros((len(classIDs), embedding_size))
        for i, classID in enumerate(classIDs):
            W[i,:] = classifiers[augID][classID]

        classifiers[augID] = W

    #now, it's time to do some predictions!
    print('doing zero-shot predictions...')
    A_dict = {}
    B_dict = {augID : {} for augID in sorted(aug_dict.keys()) if augID != 'noop'}
    C_dict = {augID : {} for augID in sorted(aug_dict.keys()) if augID != 'noop'}
    for classID in tqdm(classIDs):
        for image_path in tqdm(class2filenames_dict[classID]):
            image_base = os.path.basename(image_path)
            if image_base in BAD_IMAGE_BASES:
                continue

            image_embedding_unaug = image_embedding_dict[(image_base, 'noop')]
            classifier_unaug = classifiers['noop']
            A_dict[image_base] = run_pred(image_embedding_unaug, classifier_unaug, classIDs)

            for augID in sorted(aug_dict.keys()):
                if augID == 'noop':
                    continue

                image_embedding_aug = image_embedding_dict[(image_base, augID)]
                B_dict[augID][image_base] = run_pred(image_embedding_aug, classifier_unaug, classIDs)
                classifier_aug = classifiers[augID]
                C_dict[augID][image_base] = run_pred(image_embedding_aug, classifier_aug, classIDs)

    return A_dict, B_dict, C_dict

#returns 1 if any of preds[:top_k] matches gt, 0 otherwise
def is_it_correct(preds, gt, top_k):
    return int(any([pred == gt for pred in preds[:top_k]]))

#will return stats_dict, which has the stats in it
#will be keyed something like stats_dict[top_k]['primary'][augID][stat_name], stats_dict[top_k]['secondary'][whatever]
#top_k denotes the top-k accuracy
def do_statistical_analysis_zeroshot(class2filenames_dict, A_dict, B_dict, C_dict):
    big_stats_dict = {}
    for top_k in [1, 5, 10]:
        stats_dict = {'primary' : {}, 'secondary' : {}}
        for augID in tqdm(sorted(B_dict.keys())):
            assert(augID != 'noop')
            
            #gather values
            #these will be 1s for correct predictions and 0s for incorrect predictions
            A_vals = [] #yes this is redundant computation
            B_vals = []
            C_vals = []
            for classID in sorted(class2filenames_dict.keys()):
                for image_path in class2filenames_dict[classID]:
                    image_base = os.path.basename(image_path)
                    if image_base in BAD_IMAGE_BASES:
                        continue

                    A_vals.append(is_it_correct(A_dict[image_base], classID, top_k))
                    B_vals.append(is_it_correct(B_dict[augID][image_base], classID, top_k))
                    C_vals.append(is_it_correct(C_dict[augID][image_base], classID, top_k))

            #compute accuracies and shifts in accuracies
            unaugmented_acc = 100.0 * np.mean(A_vals)
            if 'unaugmented_acc_as_percentage' in stats_dict['secondary']:
                assert(unaugmented_acc == stats_dict['secondary']['unaugmented_acc_as_percentage'])

            stats_dict['secondary']['unaugmented_acc_as_percentage'] = unaugmented_acc

            B_acc = 100.0 * np.mean(B_vals)
            stats_dict['primary'][augID] = {}
            stats_dict['primary'][augID]['acc_decrease_as_percentage'] = unaugmented_acc - B_acc
            stats_dict['primary'][augID]['acc_recovery_as_percentage'] = 100.0 * np.mean(C_vals) - B_acc
        
        big_stats_dict[top_k] = stats_dict

    return big_stats_dict

#assumes that image_embedding_dict and text_embedding_dict have already been normalized
#returns the following dicts:
#-A_dict: cosine similarities for unaugmented image with standard prompt. indexed by image_base
#-B_dict: cosine similarities for augmented image with standard prompt. indexed by augID, then by image_base
#-C_dict: cosine similarities for augmented image with augmented prompt. indexed by augID, then by image_base
#-D_dict, E_dict, F_dict: analogous to A_dict, B_dict, C_dict except averaged over every class EXCEPT the ground-truth class
#all 6 of these dicts will use averaging to deal with multiple className's per classID or multiple text_aug_template's per augID
#cosine similarity shall always be calculated as a simple dot-product, as we're assuming that embeddings have already been normalized
def compute_cosine_similarities(class2filenames_dict, class2words_dict, aug_dict, image_embedding_dict, text_embedding_dict):
    A_dict = {}
    B_dict = {augID : {} for augID in sorted(aug_dict.keys()) if augID != 'noop'}
    C_dict = {augID : {} for augID in sorted(aug_dict.keys()) if augID != 'noop'}
    D_dict = {}
    E_dict = {augID : {} for augID in sorted(aug_dict.keys()) if augID != 'noop'}
    F_dict = {augID : {} for augID in sorted(aug_dict.keys()) if augID != 'noop'}

    #build (average) positive embeddings for computing A_dict, B_dict, C_dict
    #pos_text_embeddings will be indexed by augID, and then by gt classID
    pos_text_embeddings = {}
    for augID in sorted(aug_dict.keys()):
        pos_text_embeddings[augID] = {}
        for classID in sorted(class2filenames_dict.keys()):
            text_embeddings = []
            for className in class2words_dict[classID]:
                for text_aug_template in aug_dict[augID]['text_aug_templates']:
                    k = (classID, className, augID, text_aug_template)
                    text_embeddings.append(text_embedding_dict[k])

            v = np.mean(text_embeddings, axis=0)
            v = v / np.linalg.norm(v)
            pos_text_embeddings[augID][classID] = v

    #build average negative embeddings for computing D_dict, E_dict, F_dict
    #neg_text_embeddings will be indexed by augID, and then by gt classID
    neg_text_embeddings = {}
    for augID in sorted(aug_dict.keys()):
        #make lists so we can drop something into them for each "other" class
        neg_text_embeddings[augID] = {classID : [] for classID in sorted(class2filenames_dict.keys())}
        for otherClassID in sorted(class2filenames_dict.keys()):
            for classID in sorted(class2filenames_dict.keys()):
                if classID == otherClassID:
                    continue

                neg_text_embeddings[augID][classID].append(pos_text_embeddings[augID][otherClassID])

        #now average across the "other" classes
        #no need to renormalize here
        #technically we're trying to average over (image_base, otherClassID) pairs
        for classID in sorted(class2filenames_dict.keys()):
            neg_text_embeddings[augID][classID] = np.mean(neg_text_embeddings[augID][classID], axis=0)

    #gonna build all 6 dicts at once
    #iterate through each class
    for classID in tqdm(sorted(class2filenames_dict.keys())):

        #now iterate through all the images in this class
        for image_path in tqdm(class2filenames_dict[classID]):
            image_base = os.path.basename(image_path)
            if image_base in BAD_IMAGE_BASES:
                continue

            #populate A_dict/D_dict
            image_embedding_unaug = image_embedding_dict[(image_base, 'noop')]
            pos_text_embedding_unaug = pos_text_embeddings['noop'][classID]
            neg_text_embedding_unaug = neg_text_embeddings['noop'][classID]
            A_dict[image_base] = np.dot(image_embedding_unaug, pos_text_embedding_unaug)
            D_dict[image_base] = np.dot(image_embedding_unaug, neg_text_embedding_unaug)

            #populate B_dict/E_dict and C_dict/F_dict
            for augID in sorted(aug_dict.keys()):
                if augID == 'noop':
                    continue

                #B_dict/E_dict
                image_embedding_aug = image_embedding_dict[(image_base, augID)]
                B_dict[augID][image_base] = np.dot(image_embedding_aug, pos_text_embedding_unaug)
                E_dict[augID][image_base] = np.dot(image_embedding_aug, neg_text_embedding_unaug)

                #C_dict/F_dict
                pos_text_embedding_aug = pos_text_embeddings[augID][classID]
                neg_text_embedding_aug = neg_text_embeddings[augID][classID]
                C_dict[augID][image_base] = np.dot(image_embedding_aug, pos_text_embedding_aug)
                F_dict[augID][image_base] = np.dot(image_embedding_aug, neg_text_embedding_aug)

    return A_dict, B_dict, C_dict, D_dict, E_dict, F_dict

#will return stats_dict, which has the stats in it
#will be keyed something like stats_dict['primary'][augID][stat_name], stats_dict['secondary'][whatever]
def do_statistical_analysis_cossim(A_dict, B_dict, C_dict, D_dict, E_dict, F_dict):
    stats_dict = {'primary' : {}}
    for augID in tqdm(sorted(B_dict.keys())):
        assert(augID != 'noop')
        
        #gather values
        A_vals = []
        B_vals = []
        C_vals = []
        D_vals = []
        E_vals = []
        F_vals = []
        for image_base in tqdm(sorted(A_dict.keys())):
            A_vals.append(A_dict[image_base])
            B_vals.append(B_dict[augID][image_base])
            C_vals.append(C_dict[augID][image_base])
            D_vals.append(D_dict[image_base])
            E_vals.append(E_dict[augID][image_base])
            F_vals.append(F_dict[augID][image_base])

        A_vals = np.array(A_vals)
        B_vals = np.array(B_vals)
        C_vals = np.array(C_vals)
        D_vals = np.array(D_vals)
        E_vals = np.array(E_vals)
        F_vals = np.array(F_vals)

        #compute stats
        stats_dict['primary'][augID] = {}
        stats_dict['primary'][augID]['mean_decrease'] = np.mean(A_vals - B_vals)
        stats_dict['primary'][augID]['SD_decrease'] = np.std(A_vals - B_vals, ddof=1)
        _, stats_dict['primary'][augID]['p_decrease'] = ttest_rel(A_vals, B_vals, alternative='greater')
        stats_dict['primary'][augID]['mean_recovery'] = np.mean(C_vals - B_vals)
        stats_dict['primary'][augID]['SD_recovery'] = np.std(C_vals - B_vals, ddof=1)
        _, stats_dict['primary'][augID]['p_recovery'] = ttest_rel(C_vals, B_vals, alternative='greater')
        stats_dict['primary'][augID]['mean_decrease_diffclass'] = np.mean(D_vals - E_vals)
        stats_dict['primary'][augID]['mean_recovery_diffclass'] = np.mean(F_vals - E_vals)

    stats_dict['secondary'] = {}
    stats_dict['secondary']['avg_cossim_sameclass_unaug'] = np.mean([A_dict[k] for k in sorted(A_dict.keys())])
    stats_dict['secondary']['avg_cossim_diffclass_unaug'] = np.mean([D_dict[k] for k in sorted(D_dict.keys())])

    return stats_dict

def do_robustness_and_recovery_analysis(base_dir, embedding_dict_filename_prefix, stats_dict_filename):

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

    #normalize all embeddings
    print('normalizing image embeddings...')
    image_embedding_dict = normalize_entire_dict(image_embedding_dict)
    print('normalizing text embeddings...')
    text_embedding_dict = normalize_entire_dict(text_embedding_dict)

    #compute all the cosine similarities we'll need ("ensembling" stuff happens within this function)
    print('computing cosine similarities...')
    A_dict_cossim, B_dict_cossim, C_dict_cossim, D_dict_cossim, E_dict_cossim, F_dict_cossim = compute_cosine_similarities(class2filenames_dict, class2words_dict, aug_dict, image_embedding_dict, text_embedding_dict)

    #compute stats!
    print('computing stats (cossim)...')
    stats_dict_cossim = do_statistical_analysis_cossim(A_dict_cossim, B_dict_cossim, C_dict_cossim, D_dict_cossim, E_dict_cossim, F_dict_cossim)

    #compute zero-shot predictions
    print('computing zeroshot predictions...')
    A_dict_zeroshot, B_dict_zeroshot, C_dict_zeroshot = compute_zeroshot_preds(class2filenames_dict, class2words_dict, aug_dict, image_embedding_dict, text_embedding_dict)

    #compute stats!
    print('computing stats (zeroshot)...')
    stats_dict_zeroshot = do_statistical_analysis_zeroshot(class2filenames_dict, A_dict_zeroshot, B_dict_zeroshot, C_dict_zeroshot)

    stats_dict = {'cossim' : stats_dict_cossim, 'zeroshot' : stats_dict_zeroshot}

    with open(stats_dict_filename, 'wb') as f:
        pickle.dump(stats_dict, f)

def usage():
    print('Usage: python do_robustness_and_recovery_analysis.py <base_dir> <embedding_dict_filename_prefix> <stats_dict_filename>')

if __name__ == '__main__':
    do_robustness_and_recovery_analysis(*(sys.argv[1:]))
