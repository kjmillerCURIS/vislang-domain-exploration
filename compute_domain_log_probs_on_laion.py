import os
import sys
import numpy as np
import pickle
import shutil
import torch
import torch.nn as nn
from tqdm import tqdm
from laion_image_embedding_dataset import LaionImageEmbeddingDataset
from train_domain_classifier import load_model_and_domain_names
from compute_CLIP_embeddings import write_to_log_file
from sample_from_laion import load_log_probs_dict #yes, I know that sample_from_laion.py gets run *after* compute_log_probs_on_laion.py, but...
from experiment_params.balance_params import grab_params
from experiment_params.param_utils import get_params_key

EMBEDDING_SIZE = 768
BATCH_SIZE = 16384
SHARD_SIZE = 5000000

def compute_domain_log_probs_on_laion(experiment_dir, laion_base_dir):
    experiment_dir = os.path.abspath(os.path.expanduser(experiment_dir))
    laion_base_dir = os.path.abspath(os.path.expanduser(laion_base_dir))

    p = grab_params(get_params_key(experiment_dir))
    assert(p.domain_classifier_inference_modality == 'image')

    log_probs_dict, max_shard_index = load_log_probs_dict(experiment_dir)
    write_to_log_file('max_shard_index = %d'%(max_shard_index))
    dataset = LaionImageEmbeddingDataset(laion_base_dir, already_seen=set(log_probs_dict.keys()))
    write_to_log_file('kittycat dataset has %d samples!'%(len(dataset)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)
    model, _ = load_model_and_domain_names(experiment_dir, EMBEDDING_SIZE, device='cuda')
    model = nn.DataParallel(model)
    model.eval()

    cur_shard = {}
    cur_shard_index = max_shard_index + 1
    for t, batch in enumerate(dataloader):
        with torch.no_grad():
            embeddings, idxs = batch['embedding'].to('cuda'), batch['idx'].cpu()
            image_bases = dataset.get_image_bases(idxs)
            write_to_log_file('purr! t=%d'%(t))
            logits = model(embeddings)
            log_probs = (logits - torch.logsumexp(logits, dim=1, keepdim=True)).cpu().numpy()
            for image_base, log_probs_row in zip(image_bases, log_probs):
                cur_shard[image_base] = log_probs_row

            if len(cur_shard) >= SHARD_SIZE:
                shard_filename = os.path.join(experiment_dir, 'laion_log_probs_dict-%09d.pkl'%(cur_shard_index))
                assert(not os.path.exists(shard_filename))
                with open(shard_filename, 'wb') as f:
                    pickle.dump(cur_shard, f)
                    cur_shard = {}
                    cur_shard_index += 1

    shard_filename = os.path.join(experiment_dir, 'laion_log_probs_dict-%09d.pkl'%(cur_shard_index))
    assert(not os.path.exists(shard_filename))
    with open(shard_filename, 'wb') as f:
        pickle.dump(cur_shard, f)

def usage():
    print('Usage: python compute_domain_log_probs_on_laion.py <experiment_dir> <laion_base_dir>')

if __name__ == '__main__':
    compute_domain_log_probs_on_laion(*(sys.argv[1:]))
