import os
import sys
import numpy as np
import pickle
import torch
from tqdm import tqdm
from corrupted_cifar10_input_datasets import CorruptedCIFAR10ImageInputGroupFilteredDataset, CorruptedCIFAR10TextInputGroupFilteredDataset
from clip_adapter_model import CLIPAdapterModel
from do_subspace_analysis import do_subspace_analysis_one_embedding_dict_helper
from experiment_params.param_utils import get_params_key
from experiment_params.corrupted_cifar10_params import grab_params

#do_subspace_analysis_one_embedding_dict_helper(pair2vec, embedding_type)
#__init__(self, params, image_adapter_input_dict_filename, class_domain_dict_filename, groups)
#__init__(self, params, text_adapter_input_dict_filename, groups)

NUM_WORKERS = 2
BATCH_SIZE = 256
EMBEDDING_SIZE = 512 #ViT-B/32 is 512. ViT-L/14 is 768, which you used one time for sampling LAION.
IMAGE_INTER_SIZE = 768
TEXT_INTER_SIZE = 512

def get_dataset(experiment_dir,image_adapter_input_dict_filename,text_adapter_input_dict_filename,class_domain_dict_filename,embedding_type,groups):
    p = grab_params(get_params_key(experiment_dir))
    if embedding_type == 'image':
        return CorruptedCIFAR10ImageInputGroupFilteredDataset(p, image_adapter_input_dict_filename, class_domain_dict_filename, groups)
    elif embedding_type == 'text':
        return CorruptedCIFAR10TextInputGroupFilteredDataset(p, text_adapter_input_dict_filename, groups)
    else:
        assert(False)

def get_checkpoint_filename(experiment_dir, split_type, split_index):
    p = grab_params(get_params_key(experiment_dir))
    if split_type == 'trivial' or not p.do_disentanglement:
        return os.path.join(experiment_dir, 'checkpoints', 'checkpoint-FINAL.pth')
    else:
        assert(split_type in ['hard_zeroshot', 'easy_zeroshot'])
        return os.path.join(experiment_dir, 'checkpoints-%s-%d'%(split_type, split_index), 'checkpoint-FINAL.pth')

def get_model(experiment_dir, split_type, split_index, embedding_type):
    p = grab_params(get_params_key(experiment_dir))
    checkpoint_filename = get_checkpoint_filename(experiment_dir, split_type, split_index)
    checkpoint = torch.load(checkpoint_filename)
    if embedding_type == 'image':
        model = CLIPAdapterModel(p, EMBEDDING_SIZE, IMAGE_INTER_SIZE).to('cuda')
        model.load_state_dict(checkpoint['model_state_dict']['main']['image'])
    elif embedding_type == 'text':
        model = CLIPAdapterModel(p, EMBEDDING_SIZE, TEXT_INTER_SIZE).to('cuda')
        model.load_state_dict(checkpoint['model_state_dict']['main']['text'])
    else:
        assert(False)

    return model

def compute_pair2vec(dataset, model, embedding_type):
    pair2vec = {}
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
    for batch in tqdm(dataloader):
        Xa = batch['input'].to('cuda')
        Xb = batch['embedding'].to('cuda')
        with torch.no_grad():
            embeddings = model((Xa, Xb))
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            embeddings = embeddings.cpu().numpy()

        classes = dataset.get_classes(batch['idx'])
        domains = dataset.get_domains(batch['idx'])
        for classID, domain, embedding in zip(classes, domains, embeddings):
            k = (classID, domain)
            if embedding_type == 'image':
                if k not in pair2vec:
                    pair2vec[k] = []

                pair2vec[k].append(embedding)
            elif embedding_type == 'text':
                assert(k not in pair2vec)
                pair2vec[k] = embedding

    return pair2vec

def do_corrupted_cifar10_subspace_analysis_helper(experiment_dir, splits, split_type, split_index, seen_type, image_adapter_input_dict_filename, text_adapter_input_dict_filename, class_domain_dict_filename, embedding_type):
    if split_type == 'trivial':
        groups = splits[split_type][{'seen_groups' : 'train', 'unseen_groups' : 'test'}[seen_type]]
    else:
        assert(split_type in ['hard_zeroshot', 'easy_zeroshot'])
        groups = splits[split_type][split_index][{'seen_groups' : 'train', 'unseen_groups' : 'test'}[seen_type]]

    dataset = get_dataset(experiment_dir, image_adapter_input_dict_filename, text_adapter_input_dict_filename, class_domain_dict_filename, embedding_type, groups)
    model = get_model(experiment_dir, split_type, split_index, embedding_type)
    pair2vec = compute_pair2vec(dataset, model, embedding_type)
    results = do_subspace_analysis_one_embedding_dict_helper(pair2vec, embedding_type)
    return results

#will create a pkl file "<experiment_dir>/subspace_analysis.pkl"
#This will be a dictionary with the following levels:
#-'trivial', 'hard_zeroshot-split_0', 'easy_zeroshot-split_0' (we only analyze the subspace of split 0, even though there are other splits)
#-next level has keys 'seen_groups', and 'unseen_groups' ('trivial' will only have the 'seen_groups' key)
#-next level has keys 'image' and 'text'
#-beyond that, please refer to "do_subspace_analysis_one_embedding_dict_helper()" function
#we only look at NORMALIZED embeddings here
def do_corrupted_cifar10_subspace_analysis(experiment_dir, splits_filename, image_adapter_input_dict_filename, text_adapter_input_dict_filename, class_domain_dict_filename):
    with open(splits_filename, 'rb') as f:
        splits = pickle.load(f)

    results = {}
    for (split_type, split_index) in [('trivial', -1), ('hard_zeroshot', 0), ('easy_zeroshot', 0)]:
        split_key = split_type
        if split_type != 'trivial':
            split_key = '%s-split_%d'%(split_type, split_index)

        results[split_key] = {}
        seen_types = ['seen_groups']
        if split_type != 'trivial':
            seen_types = ['seen_groups', 'unseen_groups']

        for seen_type in seen_types:
            results[split_key][seen_type] = {}
            for embedding_type in ['image', 'text']:
                results[split_key][seen_type][embedding_type] = do_corrupted_cifar10_subspace_analysis_helper(experiment_dir, splits, split_type, 0 , seen_type, image_adapter_input_dict_filename, text_adapter_input_dict_filename, class_domain_dict_filename, embedding_type)

    results_filename = os.path.join(experiment_dir, 'subspace_analysis.pkl')
    with open(results_filename, 'wb') as f:
        pickle.dump(results, f)

def usage():
    print('Usage: python do_corrupted_cifar10_subspace_analysis.py <experiment_dir> <splits_filename> <image_adapter_input_dict_filename> <text_adapter_input_dict_filename> <class_domain_dict_filename>')

if __name__ == '__main__':
    do_corrupted_cifar10_subspace_analysis(*(sys.argv[1:]))
