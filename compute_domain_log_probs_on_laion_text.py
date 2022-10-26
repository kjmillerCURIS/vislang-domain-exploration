import os
import sys
import glob
import numpy as np
import pickle
import torch
from tqdm import tqdm
from laion_text_embedding_shard_dataset import LaionTextEmbeddingShardDataset
from compute_CLIP_embeddings import write_to_log_file
from experiment_params.balance_params import grab_params
from experiment_params.param_utils import get_params_key
from train_domain_classifier import load_model_and_domain_names

EMBEDDING_SIZE = 768
BATCH_SIZE = 4096
NUM_WORKERS = 2

def process_one_shard(text_embedding_shard_filename, model, experiment_dir):
    index_part = os.path.splitext(os.path.basename(text_embedding_shard_filename))[0].split('-')[-1]
    out_filename = os.path.join(experiment_dir, 'laion_log_probs_dict-' + index_part + '.pkl')
    if os.path.exists(out_filename):
        print('out shard file "%s" already exists, skipping computation'%(out_filename))
        return

    dataset = LaionTextEmbeddingShardDataset(text_embedding_shard_filename)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)

    out_shard = {}
    for batch in tqdm(dataloader):
        with torch.no_grad():
            embeddings = batch['text_embedding'].cuda()
            image_bases = dataset.get_image_bases(batch['idx'])
            logits = model(embeddings)
            log_probs = (logits - torch.logsumexp(logits, dim=1, keepdim=True)).cpu().numpy()
            for image_base, log_probs_row in zip(image_bases, log_probs):
                out_shard[image_base] = log_probs_row

        write_to_log_file('purrpurr! %d'%(len(out_shard)))

    with open(out_filename, 'wb') as f:
        pickle.dump(out_shard, f)

def compute_domain_log_probs_on_laion_text(experiment_dir, laion_base_dir, start_index, stride):
    experiment_dir = os.path.abspath(os.path.expanduser(experiment_dir))
    laion_base_dir = os.path.abspath(os.path.expanduser(laion_base_dir))
    start_index = int(start_index)
    stride = int(stride)

    p = grab_params(get_params_key(experiment_dir))
    assert(p.domain_classifier_inference_modality == 'text')

    #load domain-classifier model
    model, _ = load_model_and_domain_names(experiment_dir, EMBEDDING_SIZE, device='cuda')
    model.eval()

    #go through shards
    shard_filenames = sorted(glob.glob(os.path.join(laion_base_dir, 'text_embedding_dict-*.pkl')))
    for shard_filename in tqdm(shard_filenames[start_index::stride]):
        process_one_shard(shard_filename, model, experiment_dir)

def usage():
    print('Usage: python compute_domain_log_probs_on_laion_text.py <experiment_dir> <laion_base_dir> <start_index> <stride>')

if __name__ == '__main__':
    compute_domain_log_probs_on_laion_text(*(sys.argv[1:]))
