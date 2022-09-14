import os
import sys
import numpy as np
import torch
from clip_training_utils import compute_CLIP_grad_wrt_embeddings

BATCH_SIZE = 1024
EMBEDDING_SIZE = 512
DTYPE = 'float32'
TEMPERATURE = 0.07

def memory_check_callback():
    print('Meow! The GPU has %d bytes allocated'%(torch.cuda.memory_allocated()))

def can_we_handle_contrastive():
    image_embeddings = torch.from_numpy(np.random.randn(BATCH_SIZE, EMBEDDING_SIZE).astype(DTYPE)).to('cuda')
    text_embeddings = torch.from_numpy(np.random.randn(BATCH_SIZE, EMBEDDING_SIZE).astype(DTYPE)).to('cuda')

    image_grad,text_grad,nondiff_loss = compute_CLIP_grad_wrt_embeddings(image_embeddings,text_embeddings,TEMPERATURE,memory_check_callback=memory_check_callback)

if __name__ == '__main__':
    can_we_handle_contrastive()
