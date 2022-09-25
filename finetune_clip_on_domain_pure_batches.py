import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from tqdm import tqdm
from compute_CLIP_embeddings import write_to_log
from experiment_params.balance_params import grab_params
from experiment_params.param_utils import get_params_key
from laion_image_and_text_dataset_one_domain import LaionImageAndTextDatasetOneDomain
from dataloader_multiplexer import dataloader_multiplexer
from clip_training_utils import grab_clip_backbones, add_to_backbone_gradients, add_to_backbone_gradients_smallbatch

#add_to_backbone_gradients(image_backbone,text_backbone,image_batch,text_batch,image_minibatch_size,text_minibatch_size,temperature,loss_weight)
#add_to_backbone_gradients_smallbatch(image_backbone,text_backbone,image_batch,text_batch,temperature,loss_weight)

NUM_WORKERS = 4 #number of processes for dataloaders

#if there are multiple GPUs, then return {'image' : nn.DataParallel(model['image']), 'text' : nn.DataParallel(model['text'])}
#else, just return model
def optionally_parallelize_model(model):
    if torch.cuda.device_count() > 1:
        return {backbone_type : nn.DataParallel(model[backbone_type]) for backbone_type in ['image', 'text']}
    else:
        return model

def get_checkpoint_epoch_step(checkpoint_filename):
    ss = os.path.splitext(os.path.basename(checkpoint_filename))[0].split('-')[-2:]
    return (int(ss[0]), int(ss[1]))

#returns model, optimizer, scheduler, epoch, step_within_epoch, is_on_loaded_checkpoint
#(Note: model will be dictionary mapping from 'image' or 'text' to the respective backbones)
#(Note: this is the part where we decide if we will use nn.DataParallel)
#(Note: only need one optimizer, even though it has different hyperparams for different sets of params)
def get_model_optimizer_scheduler(params, checkpoint_prefix, epoch_length):
    p = params
    assert(not os.path.exists(checkpoint_prefix + '-FINAL.pth'))

    #create model
    image_backbone, text_backbone = grab_clip_backbones(p)
    model = {'image' : image_backbone, 'text' : text_backbone}
    model = optionally_parallelize_model(model)

    #create optimizer
    exclude = lambda n, p: (p.ndim < 2 or 'bn' in n or 'ln' in n or 'bias' in n) and 'logit_scale' not in n
    include = lambda n, p: not exclude(n, p) and 'logit_scale' not in n
    named_parameters = list(model['image'].named_parameters()) + list(model['text'].named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    assert(p.clip_optimizer_type == 'AdamW')
    optimizer = optim.AdamW(
        [
            {'params': gain_or_bias_params, 'weight_decay': 0.0},
            {'params': rest_params, 'weight_decay': p.clip_weight_decay},
        ],
        lr=p.clip_learning_rate,
        betas=(p.clip_beta1, p.clip_beta2),
        eps=p.clip_epsilon,
    )

    #create scheduler
    assert(p.clip_scheduler_type == 'LinearWarmupCosineAnnealingLR')
    #the lr that we initialized the optimizer with becomes the "base_lr" for the scheduler
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, p.clip_warmup_epochs * epoch_length, p.clip_max_epochs * epoch_length)

    #find latest checkpoint and update states from it if it exists
    checkpoint_filenames = sorted(glob.glob(checkpoint_prefix + '-*-*.pth'))
    checkpoint_epoch_step_list = [get_checkpoint_epoch_step(checkpoint_filename) for checkpoint_filename in checkpoint_filenames]
    if len(checkpoint_epoch_step_list) == 0:
        return model, optimizer, scheduler, 0, 0, False
    else:
        epoch_step_filename_list = [(epoch, step, filename) for (epoch, step), filename in zip(checkpoint_epoch_step_list, checkpoint_filenames)]
        _, __, best_filename = sorted(epoch_step_filename_list, reverse=True)[0]
        checkpoint = torch.load(best_filename)
        for backbone_type in ['image', 'text']:
            model[backbone_type].load_state_dict(checkpoint['model_state_dict'][backbone_type])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        step_within_epoch = checkpoint['step_within_epoch']
        return model, optimizer, scheduler, epoch, step_within_epoch, True

#makes datasets, then wraps them in dataloaders, then makes the data_genny and returns it, along with the epoch_length
def get_data_genny(params, experiment_dir, num_domains):
    p = params
    dataloaders = []
    epoch_length = 0
    for domain_index in range(num_domains):
        dataset = LaionImageAndTextDatasetOneDomain(experiment_dir, domain_index)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=p.clip_batch_size, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
        dataloaders.append(dataloader)
        epoch_length += len(dataloader)

    data_genny = dataloader_multiplexer(dataloaders)
    return data_genny, epoch_length

def finetune_clip_on_domain_pure_batches(experiment_dir):
    params_key = get_params_key(experiment_dir)
    p = grab_params(params_key)
    with open(os.path.join(experiment_dir, 'train_domain_filter.pkl'), 'rb') as f:
        domain_names = pickle.load(f)

    assert(False)

def usage():
    print('Usage: python finetune_clip_on_domain_pure_batches.py <experiment_dir>')

if __name__ == '__main__':
    finetune_clip_on_domain_pure_batches(*(sys.argv[1:]))
