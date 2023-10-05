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
from experiment_params.param_utils import get_params_key, get_train_type
from clip_training_utils import grab_clip_backbones, add_to_backbone_gradients, add_to_backbone_gradients_smallbatch
from corrupted_cifar10_input_pair_dataset import CorruptedCIFAR10InputPairDataset
from corrupted_cifar10_disentanglement_text_dataset import CorruptedCIFAR10DisentanglementTextDataset
from experiment_params.corrupted_cifar10_params import grab_params
from clip_adapter_model import CLIPAdapterModel
from disentanglement_utils import DisentanglementModel

#from finetune_clip_on_domain_pure_batches import get_model_optimizer_scheduler, is_at_fractional_checkpoint

#__init__(self, image_base_dir, class_filter=None, domain_filter=None):
#__init__(self, params, num_classes, num_domains, embedding_size, dataset=None, backbone=None, num_workers=0)

#add_to_backbone_gradients(image_backbone,text_backbone,image_batch,text_batch,image_minibatch_size,text_minibatch_size,temperature,loss_weight)
#add_to_backbone_gradients_smallbatch(image_backbone,text_backbone,image_batch,text_batch,temperature,loss_weight)

#this is kinda hacky...
EMBEDDING_SIZE = 512 #yeah, I think later I might make this a dictionary that takes in the CLIP model version
IMAGE_INTER_SIZE = 768
TEXT_INTER_SIZE = 512
NUM_CLASSES = 10
NUM_DOMAINS = 10

def genny_fn(dataloader):
    while True:
        for batch in dataloader:
            yield batch

#get generator that produces the batches
#even though these are normal batches, it's still easier for the logic to have an infinite generator and just keep track of the epoch length
#that way, we can resume an "epoch" from mid-way
def get_data_genny(params, image_adapter_input_dict_filename, text_adapter_input_dict_filename, class_domain_dict_filename, num_workers=0):
    p = params
    dataset = CorruptedCIFAR10InputPairDataset(p, image_adapter_input_dict_filename, text_adapter_input_dict_filename, class_domain_dict_filename)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=p.clip_batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    return genny_fn(dataloader), len(dataloader)

def get_disentanglement_dataset_and_genny(params, text_adapter_input_dict_filename, splits_filename, split_type='trivial', split_index=None, num_workers=0):
    p = params
    if p.disentanglement_modality == 'text':
        dataset = CorruptedCIFAR10DisentanglementTextDataset(p,text_adapter_input_dict_filename,splits_filename,split_type=split_type,split_index=split_index)
    else:
        assert(False) #not yet supported

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(p.disentanglement_batch_size, len(dataset)), shuffle=True, drop_last=True, num_workers=num_workers)
    data_genny = genny_fn(dataloader)
    return dataset, data_genny

def save_checkpoint_disentanglement(model, optimizer, scheduler, epoch, step_within_epoch, checkpoint_prefix, telemetry, telemetry_filename, is_final=False):
    checkpoint = {'epoch':epoch, 'step_within_epoch':step_within_epoch, 'model_state_dict':{}, 'optimizer_state_dict':{}, 'scheduler_state_dict':{}}
    checkpoint['model_state_dict']['main'] = {}
    checkpoint['model_state_dict']['main']['image'] = model['main']['image'].state_dict()
    checkpoint['model_state_dict']['main']['text'] = model['main']['text'].state_dict()
    checkpoint['optimizer_state_dict']['main'] = optimizer['main'].state_dict()
    checkpoint['scheduler_state_dict']['main'] = scheduler['main'].state_dict()
    if 'disentanglement' in model:
        checkpoint['model_state_dict']['disentanglement'] = model['disentanglement'].state_dict()
        checkpoint['optimizer_state_dict']['disentanglement'] = optimizer['disentanglement'].state_dict()
        if scheduler['disentanglement'] is not None:
            checkpoint['scheduler_state_dict']['disentanglement'] = scheduler['disentanglement'].state_dict()

    if not is_final:
        torch.save(checkpoint, checkpoint_prefix + '-%03d-%09d.pth'%(epoch, step_within_epoch))
    else:
        torch.save(checkpoint, checkpoint_prefix + '-FINAL.pth')

    with open(telemetry_filename, 'wb') as f:
        pickle.dump(telemetry, f)

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

def get_model_optimizer_scheduler(params, checkpoint_prefix, epoch_length):
    p = params
    assert(not os.path.exists(checkpoint_prefix + '-FINAL.pth'))

    #create model
    image_model = CLIPAdapterModel(p, EMBEDDING_SIZE, IMAGE_INTER_SIZE)
    text_model = CLIPAdapterModel(p, EMBEDDING_SIZE, TEXT_INTER_SIZE)
    image_model = image_model.to('cuda')
    text_model = text_model.to('cuda')
    _, __, temperature = grab_clip_backbones(p.clip_model_type)
    model = {'image' : image_model, 'text' : text_model}
    model = optionally_parallelize_model(model)

    #create optimizer
    assert(p.clip_optimizer_type == 'AdamW')
    optimizer = optim.AdamW(
        [
            {'params': list(model['image'].parameters()) + list(model['text'].parameters()), 'weight_decay': p.clip_weight_decay},
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
        return model, temperature, optimizer, scheduler, 0, 0, False, None
    else:
        epoch_step_filename_list = [(epoch, step, filename) for (epoch, step), filename in zip(checkpoint_epoch_step_list, checkpoint_filenames)]
        _, __, best_filename = sorted(epoch_step_filename_list, reverse=True)[0]
        checkpoint = torch.load(best_filename)
        for backbone_type in ['image', 'text']:
            model[backbone_type].load_state_dict(checkpoint['model_state_dict']['main'][backbone_type])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict']['main'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict']['main'])
        epoch = checkpoint['epoch']
        step_within_epoch = checkpoint['step_within_epoch']
        return model, temperature, optimizer, scheduler, epoch, step_within_epoch, True, checkpoint

#if you leave disentanglement_dataset as None and it turns out there isn't an existing checkpoint, then this WILL crash!
#returns model, temperature, optimizer, scheduler, epoch, step_within_epoch, is_on_loaded_checkpoint
def get_model_optimizer_scheduler_disentanglement(params, checkpoint_prefix, epoch_length, disentanglement_dataset=None, num_workers=0):
    p = params

    #get the main CLIP stuff
    main_model, temperature, main_optimizer, main_scheduler, epoch, step_within_epoch, is_on_loaded_checkpoint, checkpoint = get_model_optimizer_scheduler(p, checkpoint_prefix, epoch_length)

    #return now if there's no disentanglement
    if not p.do_disentanglement:
        return {'main' : main_model}, temperature, {'main' : main_optimizer}, {'main' : main_scheduler}, epoch, step_within_epoch, is_on_loaded_checkpoint
    #build disentanglement model
    num_classes = len(disentanglement_dataset.classes)
    num_domains = len(disentanglement_dataset.domains)
    if checkpoint is None: #need to initialize disentanglement model from scratch
        if p.disentanglement_modality == 'text':
            backbone = main_model['text']
        else:
            assert(False)

        disentanglement_model = DisentanglementModel(p, num_classes, num_domains, EMBEDDING_SIZE, dataset=disentanglement_dataset, backbone=backbone, using_clip_adapter=True, num_workers=num_workers)
    else: #no need to initialize, will just load the state-dict
        disentanglement_model = DisentanglementModel(p, num_classes, num_domains, EMBEDDING_SIZE, dataset=None, backbone=None, using_clip_adapter=True, num_workers=num_workers)

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

    #return (both main and disentanglement)
    my_model = {'main' : main_model, 'disentanglement' : disentanglement_model}
    my_optimizer = {'main' : main_optimizer, 'disentanglement' : disentanglement_optimizer}
    my_scheduler = {'main' : main_scheduler, 'disentanglement' : disentanglement_scheduler}
    return my_model, temperature, my_optimizer, my_scheduler, epoch, step_within_epoch, is_on_loaded_checkpoint

def is_at_fractional_checkpoint(params, epoch, step_in_epoch, epoch_length):
    p = params
    for fraction in p.clip_fractional_checkpoints:
        target_epoch = int(math.floor(fraction))
        target_step_in_epoch = int(round((fraction - target_epoch) * epoch_length))
        if epoch == target_epoch and step_in_epoch == target_step_in_epoch:
            return True

    return False

def train_clip_adapter_corrupted_cifar10(experiment_dir, image_adapter_input_dict_filename, text_adapter_input_dict_filename, train_class_domain_dict_filename, splits_filename, split_type='trivial', split_index=None, num_workers=0):
    num_workers = int(num_workers)
    p = grab_params(get_params_key(experiment_dir))
    train_type = get_train_type(experiment_dir)
    assert(os.path.splitext(os.path.basename(image_adapter_input_dict_filename))[0].split('-')[-1] == 'images')
    assert(os.path.splitext(os.path.basename(image_adapter_input_dict_filename))[0].split('-')[-2] == train_type)
    assert(os.path.basename(os.path.dirname(train_class_domain_dict_filename)) == train_type)

    #checkpoint_prefix
    checkpoint_dir_suffix = ''
    if split_type in ['easy_zeroshot', 'hard_zeroshot']:
        assert(p.do_disentanglement)
        split_index = int(split_index)
        checkpoint_dir_suffix = '-%s-%d'%(split_type, split_index)
    else:
        assert(split_type == 'trivial')

    checkpoint_prefix = os.path.join(experiment_dir, 'checkpoints' + checkpoint_dir_suffix, 'checkpoint')
    os.makedirs(os.path.dirname(checkpoint_prefix), exist_ok=True)

    #get generator that produces the batches
    #even though these are normal batches, it's still easier for the logic to have an infinite generator and just keep track of the epoch length
    #that way, we can resume an "epoch" from mid-way
    write_to_log_file('making dataloder...')
    data_genny, epoch_length = get_data_genny(p, image_adapter_input_dict_filename, text_adapter_input_dict_filename, train_class_domain_dict_filename, num_workers=num_workers)

    #get generator for disentanglement loss
    #we'll basically ignore any concept of "epoch" for this, cuz there's no way it'll match up with the "main" epoch
    disentanglement_dataset = None
    if p.do_disentanglement:
        disentanglement_dataset, disentanglement_data_genny = get_disentanglement_dataset_and_genny(p, text_adapter_input_dict_filename, splits_filename, split_type=split_type, split_index=split_index, num_workers=num_workers)

    #get model, optimizer, etc., refreshing from latest checkpoint if there is one
    write_to_log_file('making/loading model...')
    model,temperature,optimizer,scheduler,start_epoch,start_step_in_epoch,is_on_loaded_checkpoint = get_model_optimizer_scheduler_disentanglement(p,checkpoint_prefix,epoch_length,disentanglement_dataset=disentanglement_dataset,num_workers=num_workers)

    #setup telemetry, and refresh from any existing telemetry
    #telemetry is saved in save_checkpoint_disentanglement(), so everything should line up
    telemetry = {'epoch_length' : epoch_length, 'train_losses' : [], 'train_losses_main' : []}
    if p.do_disentanglement:
        telemetry['train_losses_disentanglement'] = []

    if p.do_closeness_loss:
        telemetry['train_losses_closeness'] = []

    telemetry_filename = os.path.join(experiment_dir, 'training_telemetry%s.pkl'%(checkpoint_dir_suffix))
    if os.path.exists(telemetry_filename):
        with open(telemetry_filename, 'rb') as f:
            telemetry = pickle.load(f)

    #main training loop
    #our convention is that we checkpoint at the BEGINNING of each epoch, e.g. the checkpoint for epoch0 would be the same as the pretrained model
    #this matches with the fractional checkpoints because the epoch checkpoints are just like 0.0, 1.0, 2.0, etc.
    #we're just saying that that many epochs have passed
    model['main']['image'].train()
    model['main']['text'].train()
    if p.do_disentanglement:
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
            image_batch = (batch['image_input'].cuda(), batch['image_embedding'].cuda())
            text_batch = (batch['text_input'].cuda(), batch['text_embedding'].cuda())
            if p.do_disentanglement:
                disentanglement_batch = next(disentanglement_data_genny)
                disentanglement_X = (disentanglement_batch['input'].cuda(), disentanglement_batch['embedding'].cuda())
                disentanglement_classes = disentanglement_batch['class'].cuda()
                disentanglement_domains = disentanglement_batch['domain'].cuda()

            optimizer['main'].zero_grad()
            if p.do_disentanglement:
                optimizer['disentanglement'].zero_grad()

            if p.clip_oversize_batch_mode:
                loss_main = add_to_backbone_gradients(model['main']['image'], model['main']['text'], image_batch, text_batch, p.clip_image_minibatch_size, p.clip_text_minibatch_size, temperature, 1.0)
            else:
                loss_main = add_to_backbone_gradients_smallbatch(model['main']['image'], model['main']['text'], image_batch, text_batch, temperature, 1.0)

            telemetry['train_losses_main'].append(loss_main.item())

            loss = loss_main
            if p.do_disentanglement:
                if p.disentanglement_modality == 'text':
                    disentanglement_embeddings = model['main']['text'](disentanglement_X)
                else:
                    assert(False)

                loss_disentanglement = model['disentanglement'](disentanglement_embeddings, disentanglement_classes, disentanglement_domains)
                telemetry['train_losses_disentanglement'].append(loss_disentanglement.item())
                loss = loss + p.disentanglement_lambda * loss_disentanglement

            if p.do_closeness_loss:
                image_embeddings = model['main']['image'](image_batch)
                text_embeddings = model['main']['text'](text_batch)
                loss_closeness = torch.mean(torch.sum(torch.square(image_embeddings / image_embeddings.norm(dim=1, keepdim=True) - text_embeddings / text_embeddings.norm(dim=1, keepdim=True)), dim=1))
                telemetry['train_losses_closeness'].append(loss_closeness.item())
                loss = loss + p.closeness_lambda * loss_closeness

            telemetry['train_losses'].append(loss.item())

            #in practice, this will only backprop through loss_disentanglement, because loss_main is detached
            #this is a good thing, because "add_to_backbone_gradients" already did the necessary backprop there
            #but it's good to have this line, because in the future someone might do loss_main the normal way
            #in which case you'd want this line to backprop through loss_main
            should_call_backward = p.do_disentanglement or p.do_closeness_loss #change this if loss_main actually needs a backward() call
            if should_call_backward:
                loss.backward()

            optimizer['main'].step()
            if p.do_disentanglement:
                optimizer['disentanglement'].step()

            scheduler['main'].step()
            if p.do_disentanglement and scheduler['disentanglement'] is not None:
                scheduler['disentanglement'].step()

        cur_start_step_in_epoch = 0

    #now save the final checkpoint
    save_checkpoint_disentanglement(model, optimizer, scheduler, epoch, step_in_epoch, checkpoint_prefix, telemetry, telemetry_filename, is_final=True)
    write_to_log_file('done training')

def usage():
    print('Usage: python train_clip_adapter_corrupted_cifar10.py <experiment_dir> <image_adapter_input_dict_filename> <text_adapter_input_dict_filename> <train_class_domain_dict_filename> <splits_filename> [<split_type="trivial">] [<split_index>=None] [<num_workers>=0]')

if __name__ == '__main__':
    train_clip_adapter_corrupted_cifar10(*(sys.argv[1:]))
