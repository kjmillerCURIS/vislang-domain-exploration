import os
import sys
import glob
import itertools
import numpy as np
import pickle
import torch
from tqdm import tqdm
from non_image_data_utils_corrupted_cifar10 import CLASS_NAMES
from general_aug_utils_corrupted_cifar10 import generate_aug_dict
from corrupted_cifar10_input_datasets import CorruptedCIFAR10ImageInputOneDomainDataset, CorruptedCIFAR10TextInputDataset
from clip_adapter_model import CLIPAdapterModel
from experiment_params.param_utils import get_params_key, get_train_type
from experiment_params.corrupted_cifar10_params import grab_params
from compute_CLIP_embeddings import write_to_log_file

''' This script will compute and save predictions on each checkpoint, and also accuracy stats '''
''' It will write a separate file per checkpoint (yes, I know that's different than before, but I think this way is better) '''

NUM_WORKERS = 1
BATCH_SIZE = 256
EMBEDDING_SIZE = 512 #ViT-B/32 is 512. ViT-L/14 is 768, which you used one time for sampling LAION.
IMAGE_INTER_SIZE = 768
TEXT_INTER_SIZE = 512

#returns classifiers, which has keys 'avg_domains', 'own_domain', and 'classes'
#classifiers['avg_domains'] and classifiers['own_domain'][domain] will map to a matrix where each COLUMN is a normalized text embedding or avg of text embeddings
#classifiers['classes'] will map to a list of classIDs, in the order that they appear in the matrices
#note that each matrix will be a tensor living on the GPU, so you can directly apply it to batches of image embeddings
#(yes, we will take the text embeddings themselves off and back onto the GPU. But the image embeddings can stay on GPU.)
#to apply a classifier to a batch of embeddings, do batch @ classifier, and you'll get a bach of score profiles
#if you set swap_class_and_domain=True, then the role of class and domain will be switched, and so will naming convention of keys
def make_classifiers(test_groups, text_input_dataset, text_model, params, swap_class_and_domain=False, text_input_dataset_domainless=None):
    p = params
    assert((text_input_dataset_domainless is None and (p.domainless_text_prop == 0.0 or swap_class_and_domain)) or (text_input_dataset_domainless is not None and p.domainless_text_prop > 0.0 and not swap_class_and_domain))
    targets = sorted(set([g[0] for g in test_groups]))
    non_targets = sorted(generate_aug_dict().keys())
    key_labels = 'classes'
    key_ensemble = 'avg_domains'
    key_oracle = 'own_domain'
    if swap_class_and_domain:
        targets = sorted(set([g[1] for g in test_groups]))
        non_targets = sorted(CLASS_NAMES.keys())
        key_labels = 'domains'
        key_ensemble = 'avg_classes'
        key_oracle = 'own_class'

    use_domainless = (not swap_class_and_domain) and (p.domainless_text_prop > 0.0)
    if use_domainless:
        key_domainless = 'domainless'
        key_ensemble_domainless = 'avg_domains_plus_domainless'
        key_oracle_domainless = 'own_domain_plus_domainless'

    #setup classifiers
    classifiers = {key_labels : targets}
    classifiers[key_ensemble] = np.zeros((EMBEDDING_SIZE, len(targets)), dtype=np.float32)
    classifiers[key_oracle] = {non_target : np.zeros((EMBEDDING_SIZE, len(targets)), dtype=np.float32) for non_target in non_targets}
    sums = {}
    sums[key_ensemble] = np.zeros((1, len(targets)), dtype=np.float32)
    sums[key_oracle] = {non_target : np.zeros((1, len(targets)), dtype=np.float32) for non_target in non_targets}
    if use_domainless:
        classifiers[key_domainless] = np.zeros((EMBEDDING_SIZE, len(targets)), dtype=np.float32)
        classifiers[key_ensemble_domainless] = np.zeros((EMBEDDING_SIZE, len(targets)), dtype=np.float32)
        classifiers[key_oracle_domainless] = {non_target : np.zeros((EMBEDDING_SIZE, len(targets)), dtype=np.float32) for non_target in non_targets}
        sums[key_domainless] = np.zeros((1, len(targets)), dtype=np.float32)

    #populate (not domainless)
    dataloader = torch.utils.data.DataLoader(text_input_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
    for batch in tqdm(dataloader):
        batch_targets = text_input_dataset.get_classes(batch['idx'])
        batch_non_targets = text_input_dataset.get_domains(batch['idx'])
        if swap_class_and_domain:
            batch_targets, batch_non_targets = batch_non_targets, batch_targets

        Xa = batch['text_input'].to('cuda')
        Xb = batch['text_embedding'].to('cuda')
        with torch.no_grad():
            embeddings = text_model((Xa, Xb))
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            embeddings = embeddings.cpu().numpy()

        for target, non_target, embedding in zip(batch_targets, batch_non_targets, embeddings):
            if target not in targets:
                continue

            assert(non_target in non_targets)
            i = targets.index(target)
            classifiers[key_ensemble][:,i] += embedding
            sums[key_ensemble][0,i] += 1
            classifiers[key_oracle][non_target][:,i] += embedding
            sums[key_oracle][non_target][0,i] += 1

    #check counts (not domainless)
    assert(np.all(sums[key_ensemble] == len(non_targets)))
    for non_target in non_targets:
        assert(np.all(sums[key_oracle][non_target] == 1))

    #divide for average (not domainless)
    with torch.no_grad():
        classifiers[key_ensemble] = torch.tensor(classifiers[key_ensemble] / sums[key_ensemble], dtype=torch.float32).to('cuda')
        for non_target in non_targets:
            classifiers[key_oracle][non_target] = torch.tensor(classifiers[key_oracle][non_target] / sums[key_oracle][non_target], dtype=torch.float32).to('cuda')

    if use_domainless:
        #populate, check, and divide domainless classifier
        dataloader = torch.utils.data.DataLoader(text_input_dataset_domainless, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
        for batch in tqdm(dataloader):
            batch_targets = text_input_dataset_domainless.get_classes(batch['idx'])
            Xa = batch['text_input'].to('cuda')
            Xb = batch['text_embedding'].to('cuda')
            with torch.no_grad():
                embeddings = text_model((Xa, Xb))
                embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embeddings = embeddings.cpu().numpy()

            for target, embedding in zip(batch_targets, embeddings):
                if target not in targets:
                    continue

                i = targets.index(target)
                classifiers[key_domainless][:,i] += embedding
                sums[key_domainless][0,i] += 1

        assert(np.all(sums[key_domainless] == 1))
        with torch.no_grad():
            classifiers[key_domainless] = torch.tensor(classifiers[key_domainless] / sums[key_domainless], dtype=torch.float32).to('cuda')

        #blend with other classifiers
        with torch.no_grad():
            classifiers[key_ensemble_domainless] = p.domainless_text_prop * classifiers[key_domainless] + (1 - p.domainless_text_prop) * classifiers[key_ensemble]
            for non_target in non_targets:
                classifiers[key_oracle_domainless][non_target] = p.domainless_text_prop * classifiers[key_domainless] + (1 - p.domainless_text_prop) * classifiers[key_oracle][non_target]

    #normalize everything
    with torch.no_grad():
        classifiers[key_ensemble] = classifiers[key_ensemble] / classifiers[key_ensemble].norm(dim=0, keepdim=True)
        for non_target in non_targets:
            classifiers[key_oracle][non_target] = classifiers[key_oracle][non_target] / classifiers[key_oracle][non_target].norm(dim=0, keepdim=True)

        if use_domainless:
            classifiers[key_domainless] = classifiers[key_domainless] / classifiers[key_domainless].norm(dim=0, keepdim=True)
            classifiers[key_ensemble_domainless] = classifiers[key_ensemble_domainless] / classifiers[key_ensemble_domainless].norm(dim=0, keepdim=True)
            for non_target in non_targets:
                classifiers[key_oracle_domainless][non_target] = classifiers[key_oracle_domainless][non_target] / classifiers[key_oracle_domainless][non_target].norm(dim=0, keepdim=True)

    return classifiers

#returns pred_dict, which has keys 'avg_domains' and 'own_domain'
#pred_dict['avg_domains'][image_base] and pred_dict['own_domain'][domain][image_base] will map to a predicted classID
#we pass in datasets because this function might be called multiple times with different models, but same datasets
#image_input_one_domain_datasets should be a dict mapping each domain to a dataset the only gives image inputs from that domain
#assume that image_model and text_model are already on GPU
#if you set swap_class_and_domain=True, then the role of class and domain will be switched, and so will naming convention of keys
#image_input_one_non_target_datasets will only give images that are in the test_groups (test_groups is just there for double-check)
def run_preds(test_groups, image_input_one_non_target_datasets, text_input_dataset, image_model, text_model, params, swap_class_and_domain=False, text_input_dataset_domainless=None):
    p = params
    assert((text_input_dataset_domainless is None and (p.domainless_text_prop == 0.0 or swap_class_and_domain)) or (text_input_dataset_domainless is not None and p.domainless_text_prop > 0.0 and not swap_class_and_domain))
    targets = sorted(set([g[0] for g in test_groups]))
    non_targets = sorted(set([g[1] for g in test_groups]))
    key_labels = 'classes'
    key_ensemble = 'avg_domains'
    key_oracle = 'own_domain'
    if swap_class_and_domain:
        targets, non_targets = non_targets, targets
        key_labels = 'domains'
        key_ensemble = 'avg_classes'
        key_oracle = 'own_class'

    use_domainless = (not swap_class_and_domain) and (p.domainless_text_prop > 0.0)
    if use_domainless:
        key_domainless = 'domainless'
        key_ensemble_domainless = 'avg_domains_plus_domainless'
        key_oracle_domainless = 'own_domain_plus_domainless'

    classifiers = make_classifiers(test_groups, text_input_dataset, text_model, p, swap_class_and_domain=swap_class_and_domain, text_input_dataset_domainless=text_input_dataset_domainless)
    write_to_log_file('made classifiers')
    pred_dict = {key_ensemble : {}, key_oracle : {non_target : {} for non_target in non_targets}}
    if use_domainless:
        pred_dict[key_domainless] = {}
        pred_dict[key_ensemble_domainless] = {}
        pred_dict[key_oracle_domainless] = {non_target : {} for non_target in non_targets}

    for non_target in tqdm(non_targets):
        dataset = image_input_one_non_target_datasets[non_target]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
        for batch in tqdm(dataloader):
            image_bases = dataset.get_image_bases(batch['idx'])
            Xa = batch['image_input'].to('cuda')
            Xb = batch['image_embedding'].to('cuda')
            batch_classes = dataset.get_classes(batch['idx'])
            batch_domains = dataset.get_domains(batch['idx'])
            for batch_classID, batch_domain in zip(batch_classes, batch_domains):
                assert((batch_classID, batch_domain) in test_groups)

            with torch.no_grad():
                embeddings = image_model((Xa, Xb))
                embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                preds_ensemble = torch.argmax(embeddings @ classifiers[key_ensemble], dim=1, keepdim=False)
                preds_ensemble = [classifiers[key_labels][y] for y in preds_ensemble.cpu().numpy()]
                preds_oracle = torch.argmax(embeddings @ classifiers[key_oracle][non_target], dim=1, keepdim=False)
                preds_oracle = [classifiers[key_labels][y] for y in preds_oracle.cpu().numpy()]
                if use_domainless:
                    preds_domainless = torch.argmax(embeddings @ classifiers[key_domainless], dim=1, keepdim=False)
                    preds_domainless = [classifiers[key_labels][y] for y in preds_domainless.cpu().numpy()]
                    preds_ensemble_domainless = torch.argmax(embeddings @ classifiers[key_ensemble_domainless], dim=1, keepdim=False)
                    preds_ensemble_domainless = [classifiers[key_labels][y] for y in preds_ensemble_domainless.cpu().numpy()]
                    preds_oracle_domainless = torch.argmax(embeddings @ classifiers[key_oracle_domainless][non_target], dim=1, keepdim=False)
                    preds_oracle_domainless = [classifiers[key_labels][y] for y in preds_oracle_domainless.cpu().numpy()]

            for image_base, pred_ensemble, pred_oracle in zip(image_bases, preds_ensemble, preds_oracle):
                pred_dict[key_ensemble][image_base] = pred_ensemble
                pred_dict[key_oracle][non_target][image_base] = pred_oracle

            if use_domainless:
                for image_base, pred_domainless, pred_ensemble_domainless, pred_oracle_domainless in zip(image_bases, preds_domainless, preds_ensemble_domainless, preds_oracle_domainless):
                    pred_dict[key_domainless][image_base] = pred_domainless
                    pred_dict[key_ensemble_domainless][image_base] = pred_ensemble_domainless
                    pred_dict[key_oracle_domainless][non_target][image_base] = pred_oracle_domainless

    return pred_dict

def reformat_test_groups(test_groups, swap_class_and_domain):
    if swap_class_and_domain:
        return [(y, x) for (x, y) in test_groups]
    else:
        return test_groups

#pred_subdict should map from image_base to prediction
#if reweighting_class_domain_dict is defined, it will be used to reweight the accuracy to make it look as if the images had a different distribution
#if expected_domain is defined, then we'll set things up to only account for one domain, and we'll double-check that there's only one domain
#this function will return a single scalar
#if you set swap_class_and_domain=True, then the role of class and domain will be switched, and so will naming convention of keys
#if expected_non_target is not None then we'll also return oracle_weight
def compute_accuracy_helper(test_groups, pred_subdict, gt_class_domain_dict, reweighting_class_domain_dict=None, expected_non_target=None, swap_class_and_domain=False):
    targets = sorted(set([g[0] for g in test_groups]))
    non_targets = sorted(set([g[1] for g in test_groups]))
    key_target = 'class'
    key_non_target = 'domain'
    if swap_class_and_domain:
        targets, non_targets = non_targets, targets
        key_target, key_non_target = key_non_target, key_target

    reformatted_test_groups = reformat_test_groups(test_groups, swap_class_and_domain)

    if expected_non_target is not None:
        targets = [target for target in targets if (target, expected_non_target) in reformatted_test_groups]

    #setup accuracy buckets and reweighting weights
    if expected_non_target is not None:
        accs = {target : 0.0 for target in targets}
        counts = {target : 0.0 for target in targets}
        if reweighting_class_domain_dict is not None:
            reweights = {target : 0.0 for target in targets}
            total = 0.0
            for image_base in sorted(reweighting_class_domain_dict.keys()):
                my_target = reweighting_class_domain_dict[image_base][key_target]
                my_non_target = reweighting_class_domain_dict[image_base][key_non_target]
                if my_target in targets and my_non_target == expected_non_target:
                    reweights[my_target] += 1
                    total += 1

            for target in targets:
                reweights[target] = reweights[target] / total

            oracle_weight = total
        else:
            oracle_weight = len(targets)

    else:
        accs = {(target, non_target) : 0.0 for (target, non_target) in reformatted_test_groups}
        counts = {(target, non_target) : 0.0 for (target, non_target) in reformatted_test_groups}
        if reweighting_class_domain_dict is not None:
            reweights = {(target, non_target) : 0.0 for (target, non_target) in reformatted_test_groups}
            total = 0.0
            for image_base in sorted(reweighting_class_domain_dict.keys()):
                my_target = reweighting_class_domain_dict[image_base][key_target]
                my_non_target = reweighting_class_domain_dict[image_base][key_non_target]
                if (my_target, my_non_target) in reformatted_test_groups:
                    reweights[(my_target, my_non_target)] += 1
                    total += 1

            for (target, non_target) in reformatted_test_groups:
                reweights[(target, non_target)] = reweights[(target, non_target)] / total

    #populate accuracy buckets
    for image_base in sorted(pred_subdict.keys()):
        target = gt_class_domain_dict[image_base][key_target]
        non_target = gt_class_domain_dict[image_base][key_non_target]
        correct = int(pred_subdict[image_base] == target)
        if expected_non_target is not None:
            assert(non_target == expected_non_target)
            k = target
        else:
            k = (target, non_target)

        accs[k] += correct
        counts[k] += 1

    #normalize to get accuracies
    for k in sorted(accs.keys()):
        accs[k] = 100.0 * accs[k] / counts[k]

    #compute total accuracy
    if reweighting_class_domain_dict is not None:
        total_acc = 0.0
        for k in sorted(accs.keys()):
            total_acc += reweights[k] * accs[k]

        if expected_non_target is not None:
            return total_acc, oracle_weight
        else:
            return total_acc
    else:
        total_acc = 0.0
        for k in sorted(accs.keys()):
            total_acc += accs[k]

        total_acc = total_acc / len(accs)

        if expected_non_target is not None:
            return total_acc, oracle_weight
        else:
            return total_acc

#returns acc_dict, which has keys 'biased_acc_as_percentage' and 'unbiased_acc_as_percentage'
#within each one, we get keys 'avg_domains', 'own_domain'[domain], and 'own_domain_averaged'
#gt_class_domain_dict is for getting the ground-truth class of the test set
#reweighting_class_domain_dict is for computing weights to reweight the accuracies to get a biased accuracy
#if you set swap_class_and_domain=True, then the role of class and domain will be switched, and so will naming convention of keys
def compute_accuracy(test_groups, pred_dict, gt_class_domain_dict, reweighting_class_domain_dict, swap_class_and_domain=False):
    key_ensemble = 'avg_domains'
    key_oracle = 'own_domain'
    if swap_class_and_domain:
        key_ensemble = 'avg_classes'
        key_oracle = 'own_class'

    use_domainless = ('domainless' in pred_dict) #this is maybe a little bit risky...
    if use_domainless:
        key_domainless = 'domainless'
        key_ensemble_domainless = 'avg_domains_plus_domainless'
        key_oracle_domainless = 'own_domain_plus_domainless'

    acc_dict = {'biased_acc_as_percentage' : {key_oracle : {}, 'oracle_weights' : {}}, 'unbiased_acc_as_percentage' : {key_oracle : {}, 'oracle_weights' : {}}}
    if use_domainless:
        acc_dict['biased_acc_as_percentage'][key_oracle_domainless] = {}
        acc_dict['unbiased_acc_as_percentage'][key_oracle_domainless] = {}

    acc_dict['biased_acc_as_percentage'][key_ensemble] = compute_accuracy_helper(test_groups, pred_dict[key_ensemble], gt_class_domain_dict, reweighting_class_domain_dict=reweighting_class_domain_dict, expected_non_target=None, swap_class_and_domain=swap_class_and_domain)
    acc_dict['unbiased_acc_as_percentage'][key_ensemble] = compute_accuracy_helper(test_groups, pred_dict[key_ensemble], gt_class_domain_dict, reweighting_class_domain_dict=None, expected_non_target=None, swap_class_and_domain=swap_class_and_domain)
    if use_domainless:
        acc_dict['biased_acc_as_percentage'][key_domainless] = compute_accuracy_helper(test_groups, pred_dict[key_domainless], gt_class_domain_dict, reweighting_class_domain_dict=reweighting_class_domain_dict, expected_non_target=None, swap_class_and_domain=swap_class_and_domain)
        acc_dict['unbiased_acc_as_percentage'][key_domainless] = compute_accuracy_helper(test_groups, pred_dict[key_domainless], gt_class_domain_dict, reweighting_class_domain_dict=None, expected_non_target=None, swap_class_and_domain=swap_class_and_domain)
        acc_dict['biased_acc_as_percentage'][key_ensemble_domainless] = compute_accuracy_helper(test_groups, pred_dict[key_ensemble_domainless], gt_class_domain_dict, reweighting_class_domain_dict=reweighting_class_domain_dict, expected_non_target=None, swap_class_and_domain=swap_class_and_domain)
        acc_dict['unbiased_acc_as_percentage'][key_ensemble_domainless] = compute_accuracy_helper(test_groups, pred_dict[key_ensemble_domainless], gt_class_domain_dict, reweighting_class_domain_dict=None, expected_non_target=None, swap_class_and_domain=swap_class_and_domain)

    for non_target in sorted(pred_dict[key_oracle].keys()):
        acc_dict['biased_acc_as_percentage'][key_oracle][non_target], acc_dict['biased_acc_as_percentage']['oracle_weights'][non_target] = compute_accuracy_helper(test_groups, pred_dict[key_oracle][non_target], gt_class_domain_dict, reweighting_class_domain_dict=reweighting_class_domain_dict, expected_non_target=non_target, swap_class_and_domain=swap_class_and_domain)
        acc_dict['unbiased_acc_as_percentage'][key_oracle][non_target], acc_dict['unbiased_acc_as_percentage']['oracle_weights'][non_target] = compute_accuracy_helper(test_groups, pred_dict[key_oracle][non_target], gt_class_domain_dict, reweighting_class_domain_dict=None, expected_non_target=non_target, swap_class_and_domain=swap_class_and_domain)
        if use_domainless:
            acc_dict['biased_acc_as_percentage'][key_oracle_domainless][non_target], _ = compute_accuracy_helper(test_groups, pred_dict[key_oracle_domainless][non_target], gt_class_domain_dict, reweighting_class_domain_dict=reweighting_class_domain_dict, expected_non_target=non_target, swap_class_and_domain=swap_class_and_domain)
            acc_dict['unbiased_acc_as_percentage'][key_oracle_domainless][non_target], _ = compute_accuracy_helper(test_groups, pred_dict[key_oracle_domainless][non_target], gt_class_domain_dict, reweighting_class_domain_dict=None, expected_non_target=non_target, swap_class_and_domain=swap_class_and_domain)

    return acc_dict

#returns result which is dict and has keys 'pred_dict' and 'acc_dict'
#if you set swap_class_and_domain=True, then the role of class and domain will be switched, and so will naming convention of keys
def evaluate_one_checkpoint(test_groups, image_input_one_non_target_datasets, text_input_dataset, image_model, text_model, params, gt_class_domain_dict, reweighting_class_domain_dict, swap_class_and_domain=False, text_input_dataset_domainless=None):
    p = params
    pred_dict = run_preds(test_groups, image_input_one_non_target_datasets, text_input_dataset, image_model, text_model, p, swap_class_and_domain=swap_class_and_domain, text_input_dataset_domainless=text_input_dataset_domainless)
    write_to_log_file('ran preds')
    acc_dict = compute_accuracy(test_groups,pred_dict,gt_class_domain_dict,reweighting_class_domain_dict,swap_class_and_domain=swap_class_and_domain)
    write_to_log_file('computed accuracies')
    result = {'pred_dict' : pred_dict, 'acc_dict' : acc_dict}
    return result

#returns image_model, text_model, kv
#image_model and text_model will be on GPU
#kv will be dictionary mapping from keys to values. These will be added to result. Will probably be things like epoch, step, etc.
def load_model(params, checkpoint_filename):
    p = params
    image_model = CLIPAdapterModel(p, EMBEDDING_SIZE, IMAGE_INTER_SIZE).to('cuda')
    text_model = CLIPAdapterModel(p, EMBEDDING_SIZE, TEXT_INTER_SIZE).to('cuda')
    checkpoint = torch.load(checkpoint_filename)
    image_model.load_state_dict(checkpoint['model_state_dict']['main']['image'])
    text_model.load_state_dict(checkpoint['model_state_dict']['main']['text'])
    kv = {'epoch' : checkpoint['epoch'], 'step_within_epoch' : checkpoint['step_within_epoch']}
    return image_model, text_model, kv

def make_result_basename(checkpoint_filename, swap_class_and_domain=False):
    if swap_class_and_domain:
        return 'result-predict_domain-' + '-'.join(os.path.splitext(os.path.basename(checkpoint_filename))[0].split('-')[1:]) + '.pkl'
    else:
        return 'result-' + '-'.join(os.path.splitext(os.path.basename(checkpoint_filename))[0].split('-')[1:]) + '.pkl'

def get_checkpoint_dir(experiment_dir, split_type, split_index):
    p = grab_params(get_params_key(experiment_dir))
    if split_type == 'trivial' or not p.do_disentanglement:
        return os.path.join(experiment_dir, 'checkpoints')
    else:
        assert(split_type in ['easy_zeroshot', 'hard_zeroshot'])
        return os.path.join(experiment_dir, 'checkpoints-%s-%d'%(split_type, split_index))

def get_result_dir(experiment_dir, split_type, split_index):
    if split_type == 'trivial':
        return os.path.join(experiment_dir, 'results')
    else:
        assert(split_type in ['easy_zeroshot', 'hard_zeroshot'])
        return os.path.join(experiment_dir, 'results-%s-%d'%(split_type, split_index))

def evaluate_checkpoints_corrupted_cifar10(experiment_dir, splits_filename, image_adapter_input_dict_filename, text_adapter_input_dict_filename, gt_class_domain_dict_filename, reweighting_class_domain_dict_filename, swap_class_and_domain=False, split_type='trivial', split_index=None):
    swap_class_and_domain = int(swap_class_and_domain)
    if split_type != 'trivial':
        assert(split_type in ['easy_zeroshot', 'hard_zeroshot'])
        split_index = int(split_index)

    with open(splits_filename, 'rb') as f:
        splits = pickle.load(f)

    if split_type == 'trivial':
        test_groups = splits[split_type]['test']
    else:
        test_groups = splits[split_type][split_index]['test']

    non_targets = sorted(set([g[1] for g in test_groups]))
    key_ensemble = 'avg_domains'
    key_oracle = 'own_domain'
    if swap_class_and_domain:
        non_targets = sorted(set([g[0] for g in test_groups]))
        key_ensemble = 'avg_classes'
        key_oracle = 'own_class'

    checkpoint_dir = get_checkpoint_dir(experiment_dir, split_type, split_index)
    result_dir = get_result_dir(experiment_dir, split_type, split_index)
    if swap_class_and_domain:
        result_dir = result_dir + '-predict_domain'

    os.makedirs(result_dir, exist_ok=True)
    p = grab_params(get_params_key(experiment_dir))

    write_to_log_file('grabbed params')

    #if I trained on unbiased, then my reweighting should be the original bias
    #otherwise, my reweighting should be exactly the same as training
    train_type = get_train_type(experiment_dir)
    if train_type == 'unbiased_train':
        assert(os.path.basename(os.path.dirname(reweighting_class_domain_dict_filename)) == 'train')
    else:
        assert(os.path.basename(os.path.dirname(reweighting_class_domain_dict_filename)) == train_type)

    image_input_one_non_target_datasets = {}
    for non_target in non_targets:
        image_input_one_non_target_datasets[non_target] = CorruptedCIFAR10ImageInputOneDomainDataset(p, image_adapter_input_dict_filename, gt_class_domain_dict_filename, non_target, test_groups, swap_class_and_domain=swap_class_and_domain)

    write_to_log_file('loaded image datasets')

    text_input_dataset = CorruptedCIFAR10TextInputDataset(p, text_adapter_input_dict_filename, domainful=True)
    text_input_dataset_domainless = None
    use_domainless = (not swap_class_and_domain) and (p.domainless_text_prop > 0.0)
    if use_domainless:
        text_input_dataset_domainless = CorruptedCIFAR10TextInputDataset(p, text_adapter_input_dict_filename, domainful=False)

    write_to_log_file('loaded text dataset')
    with open(gt_class_domain_dict_filename, 'rb') as f:
        gt_class_domain_dict = pickle.load(f)

    with open(reweighting_class_domain_dict_filename, 'rb') as f:
        reweighting_class_domain_dict = pickle.load(f)

    checkpoint_filenames = sorted(glob.glob(os.path.join(checkpoint_dir, '*.pth')))
    for checkpoint_filename in tqdm(checkpoint_filenames):
        image_model, text_model, kv = load_model(p, checkpoint_filename)
        write_to_log_file('loaded model')
        result = evaluate_one_checkpoint(test_groups, image_input_one_non_target_datasets, text_input_dataset, image_model, text_model, p, gt_class_domain_dict, reweighting_class_domain_dict, swap_class_and_domain=swap_class_and_domain, text_input_dataset_domainless=text_input_dataset_domainless)
        write_to_log_file('evaluated')
        for k in sorted(kv.keys()):
            result[k] = kv[k]

        result_filename = os.path.join(result_dir, make_result_basename(checkpoint_filename, swap_class_and_domain=swap_class_and_domain))
        with open(result_filename, 'wb') as f:
            pickle.dump(result, f)

def usage():
    print('Usage: python evaluate_checkpoints_corrupted_cifar10.py <experiment_dir> <splits_filename> <image_adapter_input_dict_filename> <text_adapter_input_dict_filename> <gt_class_domain_dict_filename> <reweighting_class_domain_dict_filename> [<swap_class_and_domain>=False] [<split_type>="trivial"] [<split_index>=None]')

if __name__ == '__main__':
    evaluate_checkpoints_corrupted_cifar10(*(sys.argv[1:]))
