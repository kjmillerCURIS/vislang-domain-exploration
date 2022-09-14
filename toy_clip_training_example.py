import os
import sys
import numpy as np
import torch
from torch import nn
import torchvision
import clip
from tqdm import tqdm
from clip_training_utils import add_to_backbone_gradients, grab_clip_backbones

#will start with (pre-trained) OpenAI CLIP backbones and train them on solid colors
#hopefully there will be room for some improvement, and hopefully that improvement will happen, showing that the training code works
