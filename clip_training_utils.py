import os
import sys
import torch
from torch import nn
import torch.nn.functional as F

#copied from https://github.com/KaiyangZhou/CoOp/blob/14a64f468d7cb976ffa7fbcce2591312c4e42aca/trainers/coop.py#L37
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

#will grab (pretrained) backbones from OpenAI CLIP model (see https://github.com/KaiyangZhou/CoOp/blob/14a64f468d7cb976ffa7fbcce2591312c4e42aca/trainers/coop.py#L37)
def grab_clip_backbones(params):
    p = params
    clip_model, _ = clip.load(p.backbone_type, device='cuda')
    image_backbone = clip_model.visual
    text_backbone = TextEncoder(clip_model)
    return image_backbone, text_backbone

#a generator
#will yield input_minibatch, start_index, end_index
def input_minibatcher(input_batch, minibatch_size):
    N = input_batch.size(dim=0)
    chunk_start = 0
    while chunk_start < N:
        chunk_end = min(chunk_start + minibatch_size, N)
        yield input_batch[chunk_start:chunk_end, :], chunk_start, chunk_end
        chunk_start = chunk_end

#gather embeddings without any autograd
#backbone sohuld be image_backbone or text_backbone from add_to_backbone_gradeints()
#ditto for input_batch, minibatch_size
def gather_embeddings_nograd(backbone, input_batch, minibatch_size):
    with torch.no_grad():
        backbone.eval()
        embeddings = []
        for input_minibatch, _, __ in input_minibatcher(input_batch, minibatch_size):
            embeddings.append(backbone(input_minibatch))

        embeddings = torch.cat(embeddings)

    return embeddings

#image_embeddings, text_embeddings will (hopefully) be NxK, where K is embedding size
#this function will return an NxN tensor
#cossims[i,j] is the similarity between image i and text j
def compute_cossims(image_embeddings, text_embeddings):

    #normalize
    image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True) #can't do "/=" with a leaf variable apparently...
    text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

    #pairwise dot-products
    cossims = image_embeddings @ text_embeddings.t()

    return cossims

#returns image_grad, text_grad, which will be same shape as image_embeddings, text_embeddings
#also returns loss, which is NOT differentiable
#memory_check_callback, if specified, will be a function with no inputs or outputs - e.g. it could print out the current memory usage
#it would be called right before compute_CLIP_grad_wrt_embeddings() returns
def compute_CLIP_grad_wrt_embeddings(image_embeddings, text_embeddings, temperature, memory_check_callback=None):

    #ensure that pytorch will compute gradients w.r.t. the embeddings
    image_embeddings.requires_grad_()
    text_embeddings.requires_grad_()

    #get cosine similarities
    cossims = compute_cossims(image_embeddings, text_embeddings)

    #get logits
    logits_per_image = cossims / temperature
    logits_per_text = logits_per_image.t()

    #compute CE loss
    labels = torch.arange(image_embeddings.size(dim=0), device=image_embeddings.device)
    loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2.0

    #backpropagate down to embeddings!
    loss.backward()

    if memory_check_callback is not None:
        memory_check_callback()

    return image_embeddings.grad, text_embeddings.grad, loss.detach()

def accumulate_into_backbone(backbone, input_batch, minibatch_size, top_grad, loss_weight):
    backbone.train()
    for input_minibatch, start_index, end_index in input_minibatcher(input_batch, minibatch_size):
        embeddings = backbone(input_minibatch)
        partial_loss = loss_weight * torch.sum(embeddings * top_grad[start_index:end_index, :]) #hopefully this loss plays nicely with DataParallel
        partial_loss.backward()

#image_backbone should be an nn.Module that can take in a batch (or minibatch) of images and output the embeddings
#if you're doing multi-GPU training, then it should already be a DataParallel
#ditto for text_backbone
#image_batch should be a batch of images, already on GPU or CPU or wherever you want it (it does NOT have to be on multiple devices if doing DataParallel)
#ditto for text_batch
#image_minibatch_size should be whatever size you think can be processed at once. It can probably be bigger if using multi-GPUs
#ditto for text_minibatch_size
#temperature is the thingy we divide the cossims by before plugging them into softmax
#loss_weight will be multiplied into whatever's added to the backbone gradients
#returns a non-differentiable loss
def add_to_backbone_gradients(image_backbone, text_backbone, image_batch, text_batch, image_minibatch_size, text_minibatch_size, temperature, loss_weight):

    orig_image_mode = image_backbone.training
    orig_text_mode = text_backbone.training
    assert(text_backbone.training == orig_mode)

    #gather image and text embeddings (without any autograd)
    image_embeddings = gather_embeddings_nograd(image_backbone, image_batch, image_minibatch_size)
    text_embeddings = gather_embeddings_nograd(text_backbone, text_batch, text_minibatch_size)

    #get gradient of loss w.r.t. embeddings
    image_grad, text_grad, nondiff_loss = compute_CLIP_grad_wrt_embeddings(image_embeddings, text_embeddings, temperature)

    #propagate the grad down the backbones
    accumulate_into_backbone(image_backbone, image_batch, image_minibatch_size, image_grad, loss_weight)
    accumulate_into_backbone(text_backbone, text_batch, text_minibatch_size, text_grad, loss_weight)

    #set back to whatever mode it was before
    image_backbone.train(orig_image_mode)
    text_backbone.train(orig_text_mode)

    return nondiff_loss

#like add_to_backbone_gradients(), but does it in one go
#this will be more efficient for small batch sizes, but it won't be able to handle large batches
#yes, I could return a loss and call "backwards", but I like having the same/similar API
def add_to_backbone_gradients_smallbatch(image_backbone, text_backbone, image_batch, text_batch, temperature, loss_weight):
    #compute embeddings
    image_embeddings = image_backbone(image_batch)
    text_embeddings = text_backbone(text_batch)

    #get cosine similarities
    cossims = compute_cossims(image_embeddings, text_embeddings)

    #get logits
    logits_per_image = cossims / temperature
    logits_per_text = logits_per_image.t()

    #compute CE loss
    labels = torch.arange(image_embeddings.size(dim=0), device=image_embeddings.device)
    loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2.0
    loss = loss_weight * loss

    #backpropagate ALL the way down!
    loss.backward()

    return loss.detach()
