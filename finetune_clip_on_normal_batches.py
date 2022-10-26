import os
import sys
import glob
import math
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from compute_CLIP_embeddings import write_to_log_file
from experiment_params.balance_params import grab_params
from experiment_params.param_utils import get_params_key
from laion_image_and_text_dataset_one_domain import LaionImageAndTextDatasetOneDomain
from clip_training_utils import grab_clip_backbones, add_to_backbone_gradients, add_to_backbone_gradients_smallbatch
from finetune_clip_on_domain_pure_batches import get_model_optimizer_scheduler, save_checkpoint, is_at_fractional_checkpoint

#add_to_backbone_gradients(image_backbone,text_backbone,image_batch,text_batch,image_minibatch_size,text_minibatch_size,temperature,loss_weight)
#add_to_backbone_gradients_smallbatch(image_backbone,text_backbone,image_batch,text_batch,temperature,loss_weight)

def genny_fn(dataloader):
    while True:
        for batch in dataloader:
            yield batch

#get generator that produces the batches
#even though these are normal batches, it's still easier for the logic to have an infinite generator and just keep track of the epoch length
#that way, we can resume an "epoch" from mid-way
def get_data_genny(params, experiment_dir, num_workers=0):
    p = params
    dataset = LaionImageAndTextDatasetOneDomain(experiment_dir, -1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=p.clip_batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    return genny_fn(dataloader), len(dataloader)

def finetune_clip_on_normal_batches(experiment_dir, num_workers=0):
    num_workers = int(num_workers)
    params_key = get_params_key(experiment_dir)
    p = grab_params(params_key)
    with open(os.path.join(experiment_dir, 'train_domain_filter.pkl'), 'rb') as f:
        domain_names = pickle.load(f)

    assert(p.clip_finetuning_batch_type == 'normal')
    assert(not p.clip_finetuning_do_disentanglement)

    #checkpoint_prefix
    checkpoint_prefix = os.path.join(experiment_dir, 'clip_finetuning_checkpoints', 'clip_finetuning_checkpoint')
    os.makedirs(os.path.dirname(checkpoint_prefix), exist_ok=True)

    #get generator that produces the batches
    #even though these are normal batches, it's still easier for the logic to have an infinite generator and just keep track of the epoch length
    #that way, we can resume an "epoch" from mid-way
    write_to_log_file('making dataloder...')
    data_genny, epoch_length = get_data_genny(p, experiment_dir, num_workers=num_workers)

    #get model, optimizer, etc., refreshing from latest checkpoint if there is one
    write_to_log_file('making/loading model...')
    model, temperature, optimizer, scheduler, start_epoch, start_step_in_epoch, is_on_loaded_checkpoint = get_model_optimizer_scheduler(p, checkpoint_prefix, epoch_length)

    #setup telemetry, and refresh from any existing telemetry
    #telemetry is saved in save_checkpoint(), so everything should line up
    telemetry = {'epoch_length' : epoch_length, 'train_losses' : []}
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
            batch = next(data_genny)
            image_batch, text_batch = batch['image'].cuda(), batch['text'].cuda()
            optimizer.zero_grad()
            if p.clip_oversize_batch_mode:
                loss = add_to_backbone_gradients(model['image'], model['text'], image_batch, text_batch, p.clip_image_minibatch_size, p.clip_text_minibatch_size, temperature, 1.0)
            else:
                loss = add_to_backbone_gradients_smallbatch(model['image'], model['text'], image_batch, text_batch, temperature, 1.0)

            telemetry['train_losses'].append(loss.item())
            optimizer.step()
            scheduler.step()

        cur_start_step_in_epoch = 0

    #now save the final checkpoint
    save_checkpoint(model, optimizer, scheduler, epoch, step_in_epoch, checkpoint_prefix, telemetry, telemetry_filename, is_final=True)
    write_to_log_file('done training')

def usage():
    print('Usage: python finetune_clip_on_normal_batches.py <experiment_dir> [<num_workers>=0]')

if __name__ == '__main__':
    finetune_clip_on_normal_batches(*(sys.argv[1:]))
