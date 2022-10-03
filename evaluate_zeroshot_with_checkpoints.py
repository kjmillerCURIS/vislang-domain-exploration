import os
import sys
import glob
import numpy as np
import pickle
from tqdm import tqdm
from experiment_params.balance_params import grab_params
from experiment_params.param_utils import get_params_key
from chunk_writer import ChunkWriter
from non_image_data_utils import load_non_image_data
from general_aug_utils import generate_aug_dict
from do_robustness_and_recovery_analysis import normalize_entire_dict, compute_zeroshot_preds, is_it_correct, BAD_IMAGE_BASES

##returns the following:
##-A_dict: maps image_base to pred_class, given unaugmented image with standard prompt
##-B_dict: maps augID, then image_base to pred_class, given augmented image with standard prompt
##-C_dict: maps augID, then image_base to pred_class, given augmented image with augmented prompt
#def compute_zeroshot_preds(class2filenames_dict, class2words_dict, aug_dict, image_embedding_dict, text_embedding_dict):
#Need to combine A_dict with B_dict to get 'standard_text_template'
#Need to combine A_dict with C_dict to get 'own_domain_text_template'

#See this code:
#            for classID in sorted(class2filenames_dict.keys()):
#                for image_path in class2filenames_dict[classID]:
#                    image_base = os.path.basename(image_path)
#                    if image_base in BAD_IMAGE_BASES:
#                        continue
#
#                    A_vals.append(is_it_correct(A_dict[image_base], classID, top_k))
#                    B_vals.append(is_it_correct(B_dict[augID][image_base], classID, top_k))
#                    C_vals.append(is_it_correct(C_dict[augID][image_base], classID, top_k))



#results[checkpoint_suffix]['epoch'/'step_within_epoch'/'epoch_length']
#results[checkpoint_suffix]['zeroshot_top1_acc_as_percentage']['standard_text_template'/'own_domain_text_template'][augID]
#save as os.path.join(experiment_dir, 'val_zeroshot_results.pkl')

#returns the 'zeroshot_top1_acc_as_percentage' part
def evaluate_zeroshot_one_checkpoint(embedding_dict_filename_prefix, val_base_dir):
    #easy dicts
    _, class2words_dict, class2filenames_dict = load_non_image_data(val_base_dir)
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

    #compute zero-shot predictions
    print('computing zeroshot predictions...')
    A_dict, B_dict, C_dict = compute_zeroshot_preds(class2filenames_dict, class2words_dict, aug_dict, image_embedding_dict, text_embedding_dict)

    #accuracies
    result_standard = {}
    result_own_domain = {}
    for standard_or_own_domain, result in zip(['standard', 'own_domain'], [result_standard, result_own_domain]):
        for augID in sorted(aug_dict.keys()):
            if augID == 'noop':
                preds_dict = A_dict
            else:
                if standard_or_own_domain == 'standard':
                    preds_dict = B_dict[augID]
                elif standard_or_own_domain == 'own_domain':
                    preds_dict = C_dict[augID]
                else:
                    assert(False)

            correctness_vals = []
            for classID in sorted(class2filenames_dict.keys()):
                for image_path in class2filenames_dict[classID]:
                    image_base = os.path.basename(image_path)
                    if image_base in BAD_IMAGE_BASES:
                        continue

                    correctness_vals.append(is_it_correct(preds_dict[image_base], classID, 1))

            result[augID] = 100.0 * np.mean(correctness_vals)

    return {'standard_text_template' : result_standard, 'own_domain_text_template' : result_own_domain}

def get_epoch_length(experiment_dir, laion_sample_size):
    p = grab_params(get_params_key(experiment_dir))
    return laion_sample_size / p.clip_batch_size

def get_epoch_and_step(embedding_dir):
    with open(os.path.join(embedding_dir, 'epoch_info_dict.pkl'), 'rb') as f:
        d = pickle.load(f)

    return d['epoch'], d['step_within_epoch']

def evaluate_zeroshot_with_checkpoints(experiment_dir, val_base_dir, laion_sample_size=2300000):
    experiment_dir = os.path.abspath(os.path.expanduser(experiment_dir))
    val_base_dir = os.path.abspath(os.path.expanduser(val_base_dir))
    laion_sample_size = int(laion_sample_size)

    epoch_length = get_epoch_length(experiment_dir, laion_sample_size)

    results_filename = os.path.join(experiment_dir, 'val_zeroshot_results.pkl')
    results = {}
    if os.path.exists(results_filename):
        with open(results_filename, 'rb') as f:
            results = pickle.load(f)

        printout_domain_averaged_results(results)

    embedding_dirs = sorted(glob.glob(os.path.join(experiment_dir, 'val_embeddings', '*')))
    for embedding_dir in tqdm(embedding_dirs):
        checkpoint_suffix = os.path.basename(embedding_dir)
        if checkpoint_suffix in results:
            print('already have "%s", skipping...'%(checkpoint_suffix))
            continue

        epoch, step_within_epoch = get_epoch_and_step(embedding_dir)
        result = {'epoch' : epoch, 'step_within_epoch' : step_within_epoch, 'epoch_length' : epoch_length}
        embedding_dict_filename_prefix = os.path.join(embedding_dir, checkpoint_suffix)
        result['zeroshot_top1_acc_as_percentage'] = evaluate_zeroshot_one_checkpoint(embedding_dict_filename_prefix, val_base_dir)
        results[checkpoint_suffix] = result
        printout_domain_averaged_results(results)
        with open(results_filename, 'wb') as f:
            pickle.dump(results, f)

    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)

    printout_domain_averaged_results(results)

#printout quick results
def printout_domain_averaged_results(results):
    print('')
    for checkpoint_suffix in sorted(results.keys()):
        print(checkpoint_suffix)
        result_standard = results[checkpoint_suffix]['zeroshot_top1_acc_as_percentage']['standard_text_template']
        print('standard_text_template: avg zeroshot_top1acc=%.1f%%'%(np.mean([result_standard[augID] for augID in sorted(result_standard.keys())])))
        result_own_domain = results[checkpoint_suffix]['zeroshot_top1_acc_as_percentage']['own_domain_text_template']
        print('own_domain_text_template: avg zeroshot_top1acc=%.1f%%'%(np.mean([result_own_domain[augID] for augID in sorted(result_own_domain.keys())])))
        print('')

def usage():
    print('Usage: python evaluate_zeroshot_with_checkpoints.py <experiment_dir> <val_base_dir> [<laion_sample_size>=2300000]')

if __name__ == '__main__':
    evaluate_zeroshot_with_checkpoints(*(sys.argv[1:]))
