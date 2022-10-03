import os
import sys
import clip
import time
import torch
from compute_CLIP_embeddings import write_to_log_file
from clip_training_utils import grab_clip_backbones, add_to_backbone_gradients_smallbatch

#add_to_backbone_gradients_smallbatch(image_backbone, text_backbone, image_batch, text_batch, temperature, loss_weight)

BATCH_SIZE_START = 2

#returns image_batch, text_batch
def generate_inputs(batch_size):
    return torch.rand((batch_size, 3, 224, 224), dtype=torch.float16).to('cuda'), clip.tokenize(['meow mix meow mix please deliver'] * batch_size).to('cuda')

def clip_GPU_stress_test():
    write_to_log_file('im running')
    image_backbone, text_backbone, temperature = grab_clip_backbones('ViT-B/32')
    write_to_log_file('grabbed backbones')
    image_backbone.train()
    text_backbone.train()
    write_to_log_file('set to train')
    batch_size = BATCH_SIZE_START
    while True:
        write_to_log_file('can we handle batch_size=%d?'%(batch_size))
        image_batch, text_batch = generate_inputs(batch_size)
        write_to_log_file('done generating inputs, now doing forwards-backwards...')
        start_time = time.time()
        add_to_backbone_gradients_smallbatch(image_backbone, text_backbone, image_batch, text_batch, temperature, 1.0)
        end_time = time.time()
        write_to_log_file(str(end_time - start_time))
        write_to_log_file('yes, we can handle batch_size=%d!'%(batch_size))
        batch_size *= 2

if __name__ == '__main__':
    clip_GPU_stress_test()
