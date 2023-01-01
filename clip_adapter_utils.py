import os
import sys
import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from compute_CLIP_embeddings import write_to_log_file

MAX_NUM_LAYERS_EXPOSED = 5

class ExposedTransformer(nn.Module):

    def __init__(self, their_transformer, num_layers_exposed):
        super().__init__()
        layers = [x for x in their_transformer.resblocks.children()]
        base_layers = []
        self.exposed_layers = []
        for t, layer in enumerate(layers):
            if t >= len(layers) - num_layers_exposed:
                self.exposed_layers.append(layer)
            else:
                base_layers.append(layer)

        self.base = nn.Sequential(*base_layers)

    #returns main_output, exposed_outputs
    def forward(self, x : torch.Tensor, cls_indices = None):
        x = self.base(x)
        exposed_outputs = {'intermediate' : []}
        for layer in self.exposed_layers:
            x = layer(x)
            x_perm = x.permute(1, 0, 2) # LND -> NLD
            if cls_indices is not None:
                x_cls = x_perm[torch.arange(x_perm.shape[0]), cls_indices]
                x_sum = torch.sum(x_perm, 1, keepdim=False, dtype=torch.float64)
                x_avg = (x_sum - x_cls.to(torch.float64)) / (1.0 * (x_perm.shape[1] - 1))
                x_avg = x_avg.to(x.dtype)
            else:
                x_cls = x_perm[:,0,:]
                x_avg = torch.mean(x_perm[:,1:,:], 1, keepdim=False, dtype=torch.float64).to(x.dtype)

            exposed_outputs['intermediate'].append({'cls' : x_cls, 'avg' : x_avg})

        return x, exposed_outputs

class ExposedVisionTransformer(nn.Module):

    def __init__(self, their_vision_transformer, num_layers_exposed):
        super().__init__()
        self.input_resolution = their_vision_transformer.input_resolution
        self.output_dim = their_vision_transformer.output_dim #not sure why this needs to be a member...
        self.conv1 = their_vision_transformer.conv1

        self.class_embedding = their_vision_transformer.class_embedding
        self.positional_embedding = their_vision_transformer.positional_embedding
        self.ln_pre = their_vision_transformer.ln_pre

        self.exposed_transformer = ExposedTransformer(their_vision_transformer.transformer, num_layers_exposed)

        self.ln_post = their_vision_transformer.ln_post
        self.proj = their_vision_transformer.proj

    def forward(self, x : torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, exposed_outputs = self.exposed_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        exposed_outputs['embedding'] = x

        return x, exposed_outputs

class ExposedTextEncoder(nn.Module):

    def __init__(self, their_clip_model, num_layers_exposed):
        super().__init__()
        self.token_embedding = their_clip_model.token_embedding
        self.exposed_transformer = ExposedTransformer(their_clip_model.transformer, num_layers_exposed)
        self.positional_embedding = their_clip_model.positional_embedding
        self.ln_final = their_clip_model.ln_final
        self.text_projection = their_clip_model.text_projection
        self.dtype = their_clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, exposed_outputs = self.exposed_transformer(x, cls_indices=text.argmax(dim=-1))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        exposed_outputs['embedding'] = x

        return x, exposed_outputs

#will grab (pretrained (and for your purposes, frozen)) backbones from OpenAI CLIP model (see https://github.com/KaiyangZhou/CoOp/blob/14a64f468d7cb976ffa7fbcce2591312c4e42aca/trainers/coop.py#L37)
#will "wrap" around those objects to expose some intermediate outputs
#using float16 for now...(not planning to finetune the backbones themselves, so in this case it doesn't really matter)
def grab_exposed_clip_backbones(clip_model_type):
    clip_model, _ = clip.load(clip_model_type, device='cuda')
    exposed_image_backbone = ExposedVisionTransformer(clip_model.visual, MAX_NUM_LAYERS_EXPOSED)
    exposed_text_backbone = ExposedTextEncoder(clip_model, MAX_NUM_LAYERS_EXPOSED)
    return exposed_image_backbone, exposed_text_backbone


#returns the thing that you'd feed into the first layer of the adapter
#returns it as a numpy array
#it's your responsibility to get the embedding for the residual part
def make_input(input_obj, num_layers_to_use):
    vecs = []
    for inter in input_obj['intermediate'][-num_layers_to_use:]:
        vecs.append(inter['cls'])
        vecs.append(inter['avg'])

    vecs.append(input_obj['embedding'])
    return np.concatenate(vecs)
