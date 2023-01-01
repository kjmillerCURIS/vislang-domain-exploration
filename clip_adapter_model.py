import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPAdapterModel(nn.Module):

    def __init__(self, params, embedding_size, inter_size):
        super().__init__()
        p = params
        layers_without_residual = []
        init_size = embedding_size + 2 * p.num_layers_to_use_for_adapter * inter_size
        cur_size = init_size
        for prop, dropout, batchnorm, activation in zip(p.adapter_hidden_layer_props, p.adapter_hidden_layer_dropouts, p.adapter_hidden_layer_batchnorms, p.adapter_hidden_layer_activations):
            next_size = int(round(prop * init_size))
            if dropout:
                layers_without_residual.append(nn.Dropout(p=dropout))

            print('HEY baselayer: (%d, %d)'%(cur_size, next_size))
            layers_without_residual.append(nn.Linear(cur_size, next_size))
            if batchnorm:
                layers_without_residual.append(nn.BatchNorm1d(next_size))

            layers_without_residual.append(eval('nn.' + activation + '()'))
            cur_size = next_size

        if p.adapter_output_layer_dropout:
            layers_without_residual.append(nn.Dropout(p=p.adapter_output_layer_dropout))

        print('HEY outlayer: (%d, %d)'%(cur_size, embedding_size))
        layers_without_residual.append(nn.Linear(cur_size, embedding_size))
        self.net_without_residual = nn.Sequential(*layers_without_residual)
        self.adapter_residual_type = p.adapter_residual_type
        if self.adapter_residual_type == 'learnable':
            self.residual_alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        else:
            assert(self.adapter_residual_type in ['fixed', 'none'])

    #inputs_and_embeddings should be the tuple (inputs, embeddings), where both inputs and embeddings are tensors
    #inputs should be EVERYTHING that you want to feed into the first layer (including a copy of the embedding probably)
    def forward(self, inputs_and_embeddings):
        (inputs, embeddings) = inputs_and_embeddings
        print('HEY inputs: %s'%(str(inputs.shape)))
        print('HEY embeddings: %s'%(str(embeddings.shape)))
        X = inputs.to(torch.float32)
        embeddings = embeddings.to(torch.float32)
        outputs = self.net_without_residual(X)
        if self.adapter_residual_type == 'learnable':
            return self.residual_alpha * outputs + (1.0 - self.residual_alpha) * embeddings
        elif self.adapter_residual_type == 'fixed':
            return 0.5 * outputs + 0.5 * embeddings
        elif self.adapter_residual_type == 'none':
            return outputs
        else:
            assert(False)
