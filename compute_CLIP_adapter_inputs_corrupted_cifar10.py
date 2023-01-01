import os
import sys
import cv2
import glob
import numpy as np
import pickle
import torch
from tqdm import tqdm
from corrupted_cifar10_raw_datasets import CorruptedCIFAR10RawImageDataset, CorruptedCIFAR10RawTextDataset
from clip_adapter_utils import grab_exposed_clip_backbones

CLIP_MODEL_TYPE = 'ViT-B/32'
BATCH_SIZE = 256
NUM_WORKERS = 2

#operates in-place, but returns exposed_outputs
def convert_to_npy(exposed_outputs):
    with torch.no_grad():
        exposed_outputs['embedding'] = exposed_outputs['embedding'].cpu().numpy()
        for tt, inter in enumerate(exposed_outputs['intermediate']):
            exposed_outputs['intermediate'][tt]['cls'] = inter['cls'].cpu().numpy()
            exposed_outputs['intermediate'][tt]['avg'] = inter['avg'].cpu().numpy()

    return exposed_outputs

#src_dir should have a basename like 'train' or 'test'
def process_images(src_dir, exposed_image_backbone, adapter_input_dict_filename):
    adapter_input_dict = {}
    dataset = CorruptedCIFAR10RawImageDataset(src_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
    for batch in tqdm(dataloader):
        with torch.no_grad():
            X = batch['image'].cuda()
            _, exposed_outputs = exposed_image_backbone(X)
            exposed_outputs = convert_to_npy(exposed_outputs)

        image_bases = dataset.get_image_bases(batch['idx'])
        for t, image_base in enumerate(image_bases):
            output = {'embedding' : exposed_outputs['embedding'][t,:], 'intermediate' : []}
            for inter in exposed_outputs['intermediate']:
                output['intermediate'].append({'cls' : inter['cls'][t,:], 'avg' : inter['avg'][t,:]})

            adapter_input_dict[image_base] = output

    with open(adapter_input_dict_filename, 'wb') as f:
        pickle.dump(adapter_input_dict, f)

def process_text(exposed_text_backbone, adapter_input_dict_filename):
    adapter_input_dict = {'domainful' : {}, 'domainless' : {}}

    #do domainful texts
    domainful_dataset = CorruptedCIFAR10RawTextDataset(domainful=True)
    dataloader = torch.utils.data.DataLoader(domainful_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
    for batch in tqdm(dataloader):
        with torch.no_grad():
            X = batch['text'].cuda()
            _, exposed_outputs = exposed_text_backbone(X)
            exposed_outputs = convert_to_npy(exposed_outputs)

        classes = domainful_dataset.get_classes(batch['idx'])
        domains = domainful_dataset.get_domains(batch['idx'])
        for t, (classID, domain) in enumerate(zip(classes, domains)):
            output = {'embedding' : exposed_outputs['embedding'][t,:], 'intermediate' : []}
            for inter in exposed_outputs['intermediate']:
                output['intermediate'].append({'cls' : inter['cls'][t,:], 'avg' : inter['avg'][t,:]})

            adapter_input_dict['domainful'][(classID, domain)] = output

    #do domainless texts
    domainless_dataset = CorruptedCIFAR10RawTextDataset(domainful=False)
    dataloader = torch.utils.data.DataLoader(domainless_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
    for batch in tqdm(dataloader):
        with torch.no_grad():
            X = batch['text'].cuda()
            _, exposed_outputs = exposed_text_backbone(X)
            exposed_outputs = convert_to_npy(exposed_outputs)

        classes = domainless_dataset.get_classes(batch['idx'])
        for t, classID in enumerate(classes):
            output = {'embedding' : exposed_outputs['embedding'][t,:], 'intermediate' : []}
            for inter in exposed_outputs['intermediate']:
                output['intermediate'].append({'cls' : inter['cls'][t,:], 'avg' : inter['avg'][t,:]})

            adapter_input_dict['domainless'][classID] = output

    #save
    with open(adapter_input_dict_filename, 'wb') as f:
        pickle.dump(adapter_input_dict, f)

def compute_CLIP_adapter_inputs_corrupted_cifar10(corrupted_cifar10_dir, adapter_input_dict_filename_prefix):
    exposed_image_backbone, exposed_text_backbone = grab_exposed_clip_backbones(CLIP_MODEL_TYPE)
    process_images(os.path.join(corrupted_cifar10_dir, 'train'), exposed_image_backbone, adapter_input_dict_filename_prefix + '-train-images.pkl')
    process_images(os.path.join(corrupted_cifar10_dir, 'test'), exposed_image_backbone, adapter_input_dict_filename_prefix + '-test-images.pkl')
    process_text(exposed_text_backbone, adapter_input_dict_filename_prefix + '-text.pkl')

def usage():
    print('Usage: python compute_CLIP_adapter_inputs_corrupted_cifar10.py <corrupted_cifar10_dir> <adapter_input_dict_filename_prefix>')

if __name__ == '__main__':
    compute_CLIP_adapter_inputs_corrupted_cifar10(*(sys.argv[1:]))
