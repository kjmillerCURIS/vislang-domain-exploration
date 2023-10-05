import os
import sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

class DisentanglementModel(nn.Module):
    ''' enforce disentanglement loss '''
    ''' takes in embeddings and measures their distance from additive model '''
    ''' maintains additive model components and adjusts them to try to fit to the data '''

    #this should work correctly as long as class and domain are independently distributed in dataset (even if the marginals aren't balanced)
    #it assumes that you want E[class_components] == 0 and E[domain_components] == 0, where expectation is w.r.t. the marginals
    #and then offset_component absorbs whatever needs to be absorbed (which in this case works out to offset_component == E[dataset])
    #no promises if they're not independent, but its probably still reasonable?
    #returns class_components (num_classes x embedding_size), domain_components (num_domains x embedding_size), offset_component (1 x embedding_size)
    #no promises about the datatype (you can assume float64 for now), so you should convert it yourself
    def __compute_initial_components(self,params,num_classes,num_domains,embedding_size,dataset,backbone,using_clip_adapter=False,num_workers=0):
        p = params
        class_components = np.zeros((num_classes, embedding_size), dtype='float64')
        domain_components = np.zeros((num_domains, embedding_size), dtype='float64')
        offset_component = np.zeros((1, embedding_size), dtype='float64')
        class_counts = np.zeros((num_classes,), dtype='float64')
        domain_counts = np.zeros((num_domains,), dtype='float64')
        total_count = 0.0
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=p.disentanglement_initialization_batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
        for batch in tqdm(dataloader):
            if using_clip_adapter:
                X = (batch['input'].cuda(), batch['embedding'].cuda())
            else:
                X = batch['X'].cuda()

            with torch.no_grad():
                embeddings = backbone(X)
                embeddings = embeddings / embeddings.norm(dim=1, keepdim=True) #normalize
                embeddings = embeddings.cpu().numpy()

            gt_classes = batch['class'].cpu().numpy()
            gt_domains = batch['domain'].cpu().numpy()
            for embedding, gt_class, gt_domain in zip(embeddings, gt_classes, gt_domains): #not a huge fan of this...
                class_components[gt_class,:] = class_components[gt_class,:] + embedding
                class_counts[gt_class] = class_counts[gt_class] + 1.0
                domain_components[gt_domain,:] = domain_components[gt_domain,:] + embedding
                domain_counts[gt_domain] = domain_counts[gt_domain] + 1.0
                offset_component = offset_component + embedding
                total_count += 1.0

        class_components = class_components / class_counts[:,np.newaxis]
        domain_components = domain_components / domain_counts[:,np.newaxis]
        offset_component = offset_component / total_count
        class_components = class_components - offset_component
        domain_components = domain_components - offset_component
        return class_components, domain_components, offset_component

    #dataset, backbone are used to initialize the components if they're not None
    #else, initialize the components with zero (which means we expect you to load them from a state dict)
    #we do NOT assume that backbone is doing any normalization
    def __init__(self, params, num_classes, num_domains, embedding_size, dataset=None, backbone=None, using_clip_adapter=False, num_workers=0):
        super().__init__()
        p = params
        self.do_disentanglement_ortho = p.do_disentanglement_ortho
        if dataset is not None and backbone is not None:
            class_components, domain_components, offset_component = self.__compute_initial_components(p, num_classes, num_domains, embedding_size, dataset, backbone, using_clip_adapter=using_clip_adapter, num_workers=num_workers)
        elif dataset is None and backbone is None:
            class_components = np.zeros((num_classes, embedding_size), dtype='float64')
            domain_components = np.zeros((num_domains, embedding_size), dtype='float64')
            offset_component = np.zeros((1, embedding_size), dtype='float64')
        else:
            assert(False)

        #use float32 instead of float16 so we don't have to worry about Adam epsilon being too small
        #in the future we might use mixed precision or just float32 for the CLIP part anyway
        self.register_parameter(name='class_components', param=nn.Parameter(torch.tensor(class_components.astype('float32'), dtype=torch.float32).requires_grad_()))
        self.register_parameter(name='domain_components', param=nn.Parameter(torch.tensor(domain_components.astype('float32'), dtype=torch.float32).requires_grad_()))
        self.register_parameter(name='offset_component', param=nn.Parameter(torch.tensor(offset_component.astype('float32'), dtype=torch.float32).requires_grad_()))


    def __compute_ortho_loss(self):
        A = self.class_components / self.class_components.norm(dim=1, keepdim=True)
        B = self.domain_components / self.domain_components.norm(dim=1, keepdim=True)
        cossims = A @ B.T
        return torch.mean(torch.square(cossims))

    #returns loss (which you'd have to multiply by a weight before calling backward() on it
    #we do NOT assume that embeddings are already normalized
    def forward(self, embeddings, gt_classes, gt_domains):
        embeddings = embeddings.float() #yes, this is differentiable!
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True) #normalize
        pred_embeddings = self.class_components[gt_classes,:] + self.domain_components[gt_domains,:] + self.offset_component
        losses = torch.sum(torch.square(pred_embeddings - embeddings), dim=1)
        loss = torch.mean(losses)
        if self.do_disentanglement_ortho:
            loss = loss + self.__compute_ortho_loss()

        return loss
