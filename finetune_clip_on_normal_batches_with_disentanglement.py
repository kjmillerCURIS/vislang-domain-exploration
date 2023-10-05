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
from finetune_clip_on_domain_pure_batches import get_model_optimizer_scheduler, is_at_fractional_checkpoint
from disentanglement_text_dataset import DisentanglementTextDataset
from disentanglement_utils import DisentanglementModel

#__init__(self, image_base_dir, class_filter=None, domain_filter=None):
#__init__(self, params, num_classes, num_domains, embedding_size, dataset=None, backbone=None, num_workers=0)

#add_to_backbone_gradients(image_backbone,text_backbone,image_batch,text_batch,image_minibatch_size,text_minibatch_size,temperature,loss_weight)
#add_to_backbone_gradients_smallbatch(image_backbone,text_backbone,image_batch,text_batch,temperature,loss_weight)

#this is kinda hacky...
NUM_CLASSES = 1000
EMBEDDING_SIZE = 512 #yeah, I think later I might make this a dictionary that takes in the CLIP model version

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

def get_disentanglement_dataset_and_genny(params, domain_names, image_base_dir, num_workers=0):
    p = params
    if p.disentanglement_modality == 'text':
        dataset = DisentanglementTextDataset(image_base_dir, domain_filter=domain_names)
    else:
        assert(False) #not yet supported

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=p.disentanglement_batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    data_genny = genny_fn(dataloader)
    return dataset, data_genny

def save_checkpoint_disentanglement(model, optimizer, scheduler, epoch, step_within_epoch, checkpoint_prefix, telemetry, telemetry_filename, is_final=False):
    checkpoint = {'epoch':epoch, 'step_within_epoch':step_within_epoch, 'model_state_dict':{}, 'optimizer_state_dict':{}, 'scheduler_state_dict':{}}
    checkpoint['model_state_dict']['main'] = {}
    checkpoint['model_state_dict']['main']['image'] = model['main']['image'].state_dict()
    checkpoint['model_state_dict']['main']['text'] = model['main']['text'].state_dict()
    checkpoint['model_state_dict']['disentanglement'] = model['disentanglement'].state_dict()
    checkpoint['optimizer_state_dict']['main'] = optimizer['main'].state_dict()
    checkpoint['optimizer_state_dict']['disentanglement'] = optimizer['disentanglement'].state_dict()
    checkpoint['scheduler_state_dict']['main'] = scheduler['main'].state_dict()
    if scheduler['disentanglement'] is not None:
        checkpoint['scheduler_state_dict']['disentanglement'] = scheduler['disentanglement'].state_dict()

    if not is_final:
        torch.save(checkpoint, checkpoint_prefix + '-%03d-%09d.pth'%(epoch, step_within_epoch))
    else:
        torch.save(checkpoint, checkpoint_prefix + '-FINAL.pth')

    with open(telemetry_filename, 'wb') as f:
        pickle.dump(telemetry, f)

#if you leave disentanglement_dataset as None and it turns out there isn't an existing checkpoint, then this WILL crash!
#returns model, temperature, optimizer, scheduler, epoch, step_within_epoch, is_on_loaded_checkpoint
def get_model_optimizer_scheduler_disentanglement(params, checkpoint_prefix, epoch_length, num_domains, disentanglement_dataset=None, num_workers=0):
    p = params

    #get the main CLIP stuff
    main_model, temperature, main_optimizer, main_scheduler, epoch, step_within_epoch, is_on_loaded_checkpoint, checkpoint = get_model_optimizer_scheduler(p, checkpoint_prefix, epoch_length, also_return_checkpoint=True, checkpoint_also_includes_disentanglement=True)

    #build disentanglement model
    if checkpoint is None: #need to initialize disentanglement model from scratch
        if p.disentanglement_modality == 'text':
            backbone = main_model['text']
        else:
            assert(False)

        disentanglement_model = DisentanglementModel(p, NUM_CLASSES, num_domains, EMBEDDING_SIZE, dataset=disentanglement_dataset, backbone=backbone, num_workers=num_workers)
    else: #no need to initialize, will just load the state-dict
        disentanglement_model = DisentanglementModel(p, NUM_CLASSES, num_domains, EMBEDDING_SIZE, dataset=None, backbone=None, num_workers=num_workers)

    disentanglement_model = disentanglement_model.to('cuda')

    #disentanglement optimizer
    if p.disentanglement_component_optimizer_type == 'Adam':
        effective_learning_rate = p.disentanglement_component_learning_rate
        if p.disentanglement_lambda > 0.0: #ensure that the components get updated at a consistent rate, regqardless of the value of lambda
            effective_learning_rate = effective_learning_rate / p.disentanglement_lambda

        disentanglement_optimizer = optim.Adam(disentanglement_model.parameters(), lr=effective_learning_rate)
    else:
        assert(False)

    #disentanglement scheduler
    if p.disentanglement_component_scheduler_type == 'none':
        disentanglement_scheduler = None
    else:
        assert(False)

    #set from checkpoint
    if checkpoint is not None:
        disentanglement_model.load_state_dict(checkpoint['model_state_dict']['disentanglement'])
        disentanglement_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['disentanglement'])
        if disentanglement_scheduler is not None:
            assert(False)
    
    model = {'main' : main_model, 'disentanglement' : disentanglement_model}
    optimizer = {'main' : main_optimizer, 'disentanglement' : disentanglement_optimizer}
    scheduler = {'main' : main_scheduler, 'disentanglement' : disentanglement_scheduler}
    return model, temperature, optimizer, scheduler, epoch, step_within_epoch, is_on_loaded_checkpoint

def finetune_clip_on_normal_batches_with_disentanglement(experiment_dir, image_base_dir, num_workers=0):
    num_workers = int(num_workers)
    params_key = get_params_key(experiment_dir)
    p = grab_params(params_key)
    with open(os.path.join(experiment_dir, 'train_domain_filter.pkl'), 'rb') as f:
        domain_names = pickle.load(f)

    assert(p.clip_finetuning_batch_type == 'normal')
    assert(p.clip_finetuning_do_disentanglement)

    #checkpoint_prefix
    checkpoint_prefix = os.path.join(experiment_dir, 'clip_finetuning_checkpoints', 'clip_finetuning_checkpoint')
    os.makedirs(os.path.dirname(checkpoint_prefix), exist_ok=True)

    #get generator that produces the batches
    #even though these are normal batches, it's still easier for the logic to have an infinite generator and just keep track of the epoch length
    #that way, we can resume an "epoch" from mid-way
    write_to_log_file('making dataloder...')
    data_genny, epoch_length = get_data_genny(p, experiment_dir, num_workers=num_workers)

    #get generator for disentanglement loss
    #we'll basically ignore any concept of "epoch" for this, cuz there's no way it'll match up with the "main" epoch
    disentanglement_dataset,disentanglement_data_genny = get_disentanglement_dataset_and_genny(p,domain_names,image_base_dir,num_workers=num_workers)

    #get model, optimizer, etc., refreshing from latest checkpoint if there is one
    write_to_log_file('making/loading model...')
    model,temperature,optimizer,scheduler,start_epoch,start_step_in_epoch,is_on_loaded_checkpoint = get_model_optimizer_scheduler_disentanglement(p,checkpoint_prefix,epoch_length,len(domain_names),disentanglement_dataset=disentanglement_dataset,num_workers=num_workers)

    #setup telemetry, and refresh from any existing telemetry
    #telemetry is saved in save_checkpoint_disentanglement(), so everything should line up
    telemetry = {'epoch_length' : epoch_length, 'train_losses' : [], 'train_losses_main' : [], 'train_losses_disentanglement' : []}
    telemetry_filename = os.path.join(experiment_dir, 'clip_finetuning_telemetry.pkl')
    if os.path.exists(telemetry_filename):
        with open(telemetry_filename, 'rb') as f:
            telemetry = pickle.load(f)

    #main training loop
    #our convention is that we checkpoint at the BEGINNING of each epoch, e.g. the checkpoint for epoch0 would be the same as the pretrained model
    #this matches with the fractional checkpoints because the epoch checkpoints are just like 0.0, 1.0, 2.0, etc.
    #we're just saying that that many epochs have passed
    model['main']['image'].train()
    model['main']['text'].train()
    model['disentanglement'].train()
    cur_start_step_in_epoch = start_step_in_epoch
    write_to_log_file('training...')
    for epoch in tqdm(range(start_epoch, p.clip_max_epochs)):
        #save a checkpoint every epoch
        if cur_start_step_in_epoch == 0 and not is_on_loaded_checkpoint:
            save_checkpoint_disentanglement(model, optimizer, scheduler, epoch, 0, checkpoint_prefix, telemetry, telemetry_filename, is_final=False)

        for step_in_epoch in tqdm(range(cur_start_step_in_epoch, epoch_length)):
            #also save any requested "fractional" checkpoints
            if is_at_fractional_checkpoint(p, epoch, step_in_epoch, epoch_length) and not is_on_loaded_checkpoint:
                save_checkpoint_disentanglement(model, optimizer, scheduler, epoch, step_in_epoch, checkpoint_prefix, telemetry, telemetry_filename, is_final=False)

            is_on_loaded_checkpoint = False

            #now time for the actual step!
            batch = next(data_genny)
            image_batch, text_batch = batch['image'].cuda(), batch['text'].cuda()
            disentanglement_batch = next(disentanglement_data_genny)
            disentanglement_X,disentanglement_classes,disentanglement_domains = disentanglement_batch['X'].cuda(),disentanglement_batch['class'].cuda(),disentanglement_batch['domain'].cuda()

            optimizer['main'].zero_grad()
            optimizer['disentanglement'].zero_grad()
            if p.clip_oversize_batch_mode:
                loss_main = add_to_backbone_gradients(model['main']['image'], model['main']['text'], image_batch, text_batch, p.clip_image_minibatch_size, p.clip_text_minibatch_size, temperature, 1.0)
            else:
                loss_main = add_to_backbone_gradients_smallbatch(model['main']['image'], model['main']['text'], image_batch, text_batch, temperature, 1.0)

            telemetry['train_losses_main'].append(loss_main.item())

            if p.disentanglement_modality == 'text':
                disentanglement_embeddings = model['main']['text'](disentanglement_X)
            else:
                assert(False)

            loss_disentanglement = model['disentanglement'](disentanglement_embeddings, disentanglement_classes, disentanglement_domains)
            loss_disentanglement = loss_disentanglement.half()
            telemetry['train_losses_disentanglement'].append(loss_disentanglement.item())
            loss = loss_main + p.disentanglement_lambda * loss_disentanglement
            telemetry['train_losses'].append(loss.item())

            #in practice, this will only backprop through loss_disentanglement, because loss_main is detached
            #this is a good thing, because "add_to_backbone_gradients" already did the necessary backprop there
            #but it's good to have this line, because in the future someone might do loss_main the normal way
            #in which case you'd want this line to backprop through loss_main
            loss.backward()

            optimizer['main'].step()
            optimizer['disentanglement'].step()
            scheduler['main'].step()
            if scheduler['disentanglement'] is not None:
                scheduler['disentanglement'].step()

        cur_start_step_in_epoch = 0

    #now save the final checkpoint
    save_checkpoint_disentanglement(model, optimizer, scheduler, epoch, step_in_epoch, checkpoint_prefix, telemetry, telemetry_filename, is_final=True)
    write_to_log_file('done training')

def usage():
    print('Usage: python finetune_clip_on_normal_batches_with_disentanglement.py <experiment_dir> <image_base_dir> [<num_workers>=0]')

if __name__ == '__main__':
    finetune_clip_on_normal_batches_with_disentanglement(*(sys.argv[1:]))
