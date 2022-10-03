import os
import sys
import glob
import math
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from tqdm import tqdm
from compute_CLIP_embeddings import write_to_log_file
from experiment_params.balance_params import grab_params
from experiment_params.param_utils import get_params_key
from laion_image_and_text_dataset_one_domain import LaionImageAndTextDatasetOneDomain
from dataloader_multiplexer import dataloader_multiplexer
from clip_training_utils import grab_clip_backbones, add_to_backbone_gradients, add_to_backbone_gradients_smallbatch

#add_to_backbone_gradients(image_backbone,text_backbone,image_batch,text_batch,image_minibatch_size,text_minibatch_size,temperature,loss_weight)
#add_to_backbone_gradients_smallbatch(image_backbone,text_backbone,image_batch,text_batch,temperature,loss_weight)

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

#returns model, temperature, optimizer, scheduler, epoch, step_within_epoch, is_on_loaded_checkpoint
#(Note: model will be dictionary mapping from 'image' or 'text' to the respective backbones)
#(Note: this is the part where we decide if we will use nn.DataParallel)
#(Note: only need one optimizer, even though it has different hyperparams for different sets of params)
def get_model_optimizer_scheduler(params, checkpoint_prefix, epoch_length):
    p = params
    assert(not os.path.exists(checkpoint_prefix + '-FINAL.pth'))

    #create model
    image_backbone, text_backbone, temperature = grab_clip_backbones(p.clip_model_type)
    model = {'image' : image_backbone, 'text' : text_backbone}
    model = optionally_parallelize_model(model)

    #create optimizer
    exclude = lambda na, pa: (pa.ndim < 2 or 'bn' in na or 'ln' in na or 'bias' in na) and 'logit_scale' not in na
    include = lambda na, pa: not exclude(na, pa) and 'logit_scale' not in na
    named_parameters = list(model['image'].named_parameters()) + list(model['text'].named_parameters())
    gain_or_bias_params = [pa for na, pa in named_parameters if exclude(na, pa) and pa.requires_grad]
    rest_params = [pa for na, pa in named_parameters if include(na, pa) and pa.requires_grad]
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
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, int(round(p.clip_warmup_epochs * epoch_length)), int(round(p.clip_max_epochs * epoch_length)))

    #find latest checkpoint and update states from it if it exists
    checkpoint_filenames = sorted(glob.glob(checkpoint_prefix + '-*-*.pth'))
    checkpoint_epoch_step_list = [get_checkpoint_epoch_step(checkpoint_filename) for checkpoint_filename in checkpoint_filenames]
    if len(checkpoint_epoch_step_list) == 0:
        return model, temperature, optimizer, scheduler, 0, 0, False
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
        return model, temperature, optimizer, scheduler, epoch, step_within_epoch, True

def save_checkpoint(model, optimizer, scheduler, epoch, step_within_epoch, checkpoint_prefix, telemetry, telemetry_filename, is_final=False):
    checkpoint = {'model_state_dict' : {'image' : model['image'].state_dict(), 'text' : model['text'].state_dict()}, 'optimizer_state_dict' : optimizer.state_dict(), 'scheduler_state_dict' : scheduler.state_dict(), 'epoch' : epoch, 'step_within_epoch' : step_within_epoch}
    if not is_final:
        torch.save(checkpoint, checkpoint_prefix + '-%03d-%09d.pth'%(epoch, step_within_epoch))
    else:
        torch.save(checkpoint, checkpoint_prefix + '-FINAL.pth')

    with open(telemetry_filename, 'wb') as f:
        pickle.dump(telemetry, f)

#makes datasets, then wraps them in dataloaders, then makes the data_genny and returns it, along with the epoch_length
def get_data_genny(params, experiment_dir, num_domains, num_workers=0):
    p = params
    dataloaders = []
    epoch_length = 0
    for domain_index in range(num_domains):
        dataset = LaionImageAndTextDatasetOneDomain(experiment_dir, domain_index)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=p.clip_batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        dataloaders.append(dataloader)
        epoch_length += len(dataloader)

    data_genny = dataloader_multiplexer(dataloaders)
    return data_genny, epoch_length

def is_at_fractional_checkpoint(params, epoch, step_in_epoch, epoch_length):
    p = params
    for fraction in p.clip_fractional_checkpoints:
        target_epoch = int(math.floor(fraction))
        target_step_in_epoch = int(round((fraction - target_epoch) * epoch_length))
        if epoch == target_epoch and step_in_epoch == target_step_in_epoch:
            return True

    return False

def finetune_clip_on_domain_pure_batches(experiment_dir, num_workers=0):
    num_workers = int(num_workers)
    params_key = get_params_key(experiment_dir)
    p = grab_params(params_key)
    with open(os.path.join(experiment_dir, 'train_domain_filter.pkl'), 'rb') as f:
        domain_names = pickle.load(f)

    #checkpoint_prefix
    checkpoint_prefix = os.path.join(experiment_dir, 'clip_finetuning_checkpoints', 'clip_finetuning_checkpoint')
    os.makedirs(os.path.dirname(checkpoint_prefix), exist_ok=True)

    #get generator that produces the batches
    write_to_log_file('making dataloders...')
    data_genny, epoch_length = get_data_genny(p, experiment_dir, len(domain_names), num_workers=num_workers)

    #get model, optimizer, etc., refreshing from latest checkpoint if there is one
    write_to_log_file('making/loading model...')
    model, temperature, optimizer, scheduler, start_epoch, start_step_in_epoch, is_on_loaded_checkpoint = get_model_optimizer_scheduler(p, checkpoint_prefix, epoch_length)

    #setup telemetry, and refresh from any existing telemetry
    #telemetry is saved in save_checkpoint(), so everything should line up
    telemetry = {'epoch_length' : epoch_length, 'train_losses' : [], 'domain_indices' : []}
    telemetry_filename = os.path.join(experiment_dir, 'clip_finetuning_telemetry.pkl')
    if os.path.exists(telemetry_filename):
        with open(telemetry_filename, 'rb') as f:
            telemetry = pickle.load(f)

    #main training loop
    #our convention is that we checkpoint at the BEGINNING of each epoch, e.g. the checkpoint for epoch0 would be the same as the pretrained model
    #this matches with the fractional checkpoints because the epoch checkpoints are just like 0.0, 1.0, 2.0, etc.
    #we're just saying that that many epochs have passed
    model['image'].train()
    model['text'].train()
    cur_start_step_in_epoch = start_step_in_epoch
    write_to_log_file('training...')
    for epoch in tqdm(range(start_epoch, p.clip_max_epochs)):
        #save a checkpoint every epoch
        if cur_start_step_in_epoch == 0 and not is_on_loaded_checkpoint:
            save_checkpoint(model, optimizer, scheduler, epoch, 0, checkpoint_prefix, telemetry, telemetry_filename, is_final=False)

        for step_in_epoch in tqdm(range(cur_start_step_in_epoch, epoch_length)):
            #also save any requested "fractional" checkpoints
            if is_at_fractional_checkpoint(p, epoch, step_in_epoch, epoch_length) and not is_on_loaded_checkpoint:
                save_checkpoint(model, optimizer, scheduler, epoch, step_in_epoch, checkpoint_prefix, telemetry, telemetry_filename, is_final=False)

            is_on_loaded_checkpoint = False

            #now time for the actual step!
            batch, domain_index = next(data_genny)
            image_batch, text_batch = batch['image'].cuda(), batch['text'].cuda()
            optimizer.zero_grad()
            if p.clip_oversize_batch_mode:
                loss = add_to_backbone_gradients(model['image'], model['text'], image_batch, text_batch, p.clip_image_minibatch_size, p.clip_text_minibatch_size, temperature, 1.0)
            else:
                loss = add_to_backbone_gradients_smallbatch(model['image'], model['text'], image_batch, text_batch, temperature, 1.0)

            telemetry['train_losses'].append(loss.item())
            telemetry['domain_indices'].append(domain_index)
            optimizer.step()
            scheduler.step()

        cur_start_step_in_epoch = 0

    #now save the final checkpoint
    save_checkpoint(model, optimizer, scheduler, epoch, step_in_epoch, checkpoint_prefix, telemetry, telemetry_filename, is_final=True)
    write_to_log_file('done training')

def usage():
    print('Usage: python finetune_clip_on_domain_pure_batches.py <experiment_dir> [<num_workers>=0]')

if __name__ == '__main__':
    finetune_clip_on_domain_pure_batches(*(sys.argv[1:]))
