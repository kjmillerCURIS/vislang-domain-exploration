import os
import sys
import glob
import math
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from experiment_params.balance_params import grab_params
from experiment_params.param_utils import validate_params_key, get_params_key
from embedding_domain_and_class_dataset import EmbeddingDomainAndClassDataset
from compute_CLIP_embeddings import write_to_log_file

class DomainClassifierModel(nn.Module):

    def __init__(self, params, embedding_size, num_domains):
        super().__init__()
        p = params
        layers_without_skip = []
        cur_size = embedding_size
        for prop, dropout, batchnorm, activation in zip(p.domain_classifier_hidden_layer_props, p.domain_classifier_hidden_layer_dropouts, p.domain_classifier_hidden_layer_batchnorms, p.domain_classifier_hidden_layer_activations):
            next_size = int(round(prop * embedding_size))
            if dropout:
                layers_without_skip.append(nn.Dropout(p=dropout))

            layers_without_skip.append(nn.Linear(cur_size, next_size))
            if batchnorm:
                layers_without_skip.append(nn.BatchNorm1d(cur_size))

            layers_without_skip.append(eval('nn.' + activation + '()'))
            cur_size = next_size

        if p.domain_classifier_output_layer_dropout:
            layers_without_skip.append(nn.Dropout(p=p.domain_classifier_output_layer_dropout))

        layers_without_skip.append(nn.Linear(cur_size, num_domains))
        self.net_without_skip = nn.Sequential(*layers_without_skip)
        if p.domain_classifier_include_skip_layer:
            self.skip_layer = nn.Linear(embedding_size, num_domains)
        else:
            self.skip_layer = None

    def forward(self, embeddings):
        X_in = embeddings
        X_in = X_in / X_in.norm(dim=1, keepdim=True)
        logits = self.net_without_skip(X_in)
        if self.skip_layer is not None:
            logits = logits + self.skip_layer(X_in)

        return logits

#not used by this script, but used by other script(s)
def load_model_and_domain_names(experiment_dir, embedding_size, device='cpu'):
    params_key = get_params_key(experiment_dir)
    p = grab_params(params_key)
    domain_names_filename = os.path.join(experiment_dir, 'train_domain_filter.pkl')
    with open(domain_names_filename, 'rb') as f:
        domain_names = pickle.load(f)

    model_filename = os.path.join(experiment_dir, 'domain_classifier_checkpoints/domain_classifier_checkpoint-FINAL.pth')
    model = DomainClassifierModel(p, embedding_size, len(domain_names))
    model = model.to(device)
    checkpoint = torch.load(model_filename, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, domain_names

def get_train_dataset(params, embedding_dict_filename_prefix, image_base_dir=None):
    p = params
    assert(p.domain_type == 'synthetic')
    assert(p.domain_split_type == 'all_all')
    assert(p.class_split_type == 'all_all')
    if p.domain_classifier_train_embedding_type == 'text':
        return EmbeddingDomainAndClassDataset(embedding_dict_filename_prefix,p.domain_classifier_train_embedding_type,domain_filter=None,class_filter=None)
    elif p.domain_classifier_train_embedding_type == 'image':
        return EmbeddingDomainAndClassDataset(embedding_dict_filename_prefix,p.domain_classifier_train_embedding_type,domain_filter=None,class_filter=None,base_dir=image_base_dir,image_shots_per_domainclass=p.domain_classifier_image_shots_per_domainclass,image_sampling_seed=p.domain_classifier_image_sampling_seed)
    else:
        assert(False)

def get_train_dataloader(params, train_dataset):
    p = params
    return torch.utils.data.DataLoader(train_dataset, batch_size=p.domain_classifier_batch_size, shuffle=True, num_workers=0)

#returns model, optimizer, scheduler, epoch, is_on_loaded_checkpoint
def get_model_optimizer_scheduler(params, checkpoint_prefix, train_dataset):
    p = params
    assert(not os.path.exists(checkpoint_prefix + '-FINAL.pth'))
    model = DomainClassifierModel(p, train_dataset.embedding_size, len(train_dataset.domain_filter))
    model = model.to('cuda')
    if p.domain_classifier_optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters())
    else:
        assert(False)

    if p.domain_classifier_scheduler == 'OneCycleLR':
        steps_per_epoch = int(math.ceil(len(train_dataset) / p.domain_classifier_batch_size))
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=p.domain_classifier_max_lr, epochs=p.domain_classifier_num_epochs, steps_per_epoch=steps_per_epoch)
    else:
        assert(False)

    checkpoint_filenames = sorted(glob.glob(checkpoint_prefix + '-*.pth'))
    checkpoint_epochs = [int(os.path.splitext(os.path.basename(s))[0].split('-')[-1]) for s in checkpoint_filenames]
    if len(checkpoint_epochs) == 0:
        return model, optimizer, scheduler, 0, False
    else:
        i_max = np.argmax(checkpoint_epochs)
        checkpoint_filename = checkpoint_filenames[i_max]
        checkpoint = torch.load(checkpoint_filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        return model, optimizer, scheduler, epoch, True

def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_prefix, is_final=False):
    checkpoint = {'model_state_dict' : model.state_dict(), 'optimizer_state_dict' : optimizer.state_dict(), 'scheduler_state_dict' : scheduler.state_dict(), 'epoch' : epoch}
    if not is_final:
        torch.save(checkpoint, checkpoint_prefix + '-%03d.pth'%(epoch))
    else:
        torch.save(checkpoint, checkpoint_prefix + '-FINAL.pth')

def do_training(params, train_dataset, experiment_dir):
    p = params

    #administrivia
    train_dataloader = get_train_dataloader(p, train_dataset)
    checkpoint_dir = os.path.join(experiment_dir, 'domain_classifier_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'domain_classifier_checkpoint')

    #create model, optimizer, scheduler, and load any previous state if needing to resume
    model, optimizer, scheduler, start_epoch, is_on_loaded_checkpoint = get_model_optimizer_scheduler(p, checkpoint_prefix, train_dataset)

    #train
    model.train()
    for epoch in tqdm(range(start_epoch, p.domain_classifier_num_epochs)):
        if epoch % p.domain_classifier_checkpoint_freq == 0 and not is_on_loaded_checkpoint:
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_prefix)

        is_on_loaded_checkpoint = False
        num_batches = 0
        train_batch_accs = []
        for batch in tqdm(train_dataloader):
            embeddings, domains = batch['embedding'], batch['domain']
            embeddings, domains = embeddings.to('cuda'), domains.to('cuda')
            optimizer.zero_grad()
            logits = model(embeddings)
            preds = torch.argmax(logits.detach(), dim=1).to('cpu').numpy()
            gts = domains.to('cpu').numpy()
            train_batch_acc = 100.0 * np.sum(preds == gts) / len(gts)
            train_batch_accs.append(train_batch_acc)
            print('train_batch_acc = %.1f%%'%(train_batch_acc))
            loss = F.cross_entropy(logits, domains)
            loss.backward()
            optimizer.step()
            scheduler.step()
            num_batches += 1

        print('train_avg_batch_acc = %.1f%%'%(np.mean(train_batch_accs)))
        print('Meow! I ran %d batches!'%(num_batches))

    save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_prefix, is_final=True)

def train_domain_classifier(params_key, embedding_dict_filename_prefix, experiment_dir, image_base_dir=None):
    embedding_dict_filename_prefix = os.path.abspath(os.path.expanduser(embedding_dict_filename_prefix))
    experiment_dir = os.path.abspath(os.path.expanduser(experiment_dir))
    if image_base_dir is not None:
        image_base_dir = os.path.abspath(os.path.expanduser(image_base_dir))

    os.makedirs(experiment_dir, exist_ok=True)

    validate_params_key(experiment_dir, params_key)
    p = grab_params(params_key)
    assert(p.sampling_method == 'classifier')

    write_to_log_file('getting train dataset...')
    train_dataset = get_train_dataset(p, embedding_dict_filename_prefix, image_base_dir=image_base_dir)
    write_to_log_file('done getting train dataset')
    with open(os.path.join(experiment_dir, 'train_domain_filter.pkl'), 'wb') as f:
        pickle.dump(train_dataset.domain_filter, f)

    do_training(p, train_dataset, experiment_dir)

def usage():
    print('Usage: python train_domain_classifier.py <params_key> <embedding_dict_filename_prefix> <experiment_dir> [<image_base_dir>=None]')

if __name__ == '__main__':
    train_domain_classifier(*(sys.argv[1:]))
