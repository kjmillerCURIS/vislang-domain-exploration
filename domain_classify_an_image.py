import os
import sys
import cv2
cv2.setNumThreads(1)
import numpy as np
import pickle
from PIL import Image
import torch
import torch.nn as nn
import clip
from experiment_params.param_utils import get_params_key
from experiment_params.balance_params import grab_params
from train_domain_classifier import load_model_and_domain_names

EMBEDDING_SIZE = 768

def domain_classify_an_image(experiment_dir, image):
    experiment_dir = os.path.expanduser(experiment_dir)
    image = os.path.expanduser(image)

    model, domain_names = load_model_and_domain_names(experiment_dir, EMBEDDING_SIZE)
    model.eval()
    clip_model, preprocess = clip.load('ViT-L/14', device='cpu')
    img = preprocess(Image.open(image)).unsqueeze(0).to('cpu')
    with torch.no_grad():
        img_feats = clip_model.encode_image(img)
        logits = model(img_feats) #model will do its own normalization
        probs = nn.Softmax(dim=1)(logits)
        probs = np.squeeze(probs.to('cpu').numpy())

    i_max = np.argmax(probs)
    for i, (domain_name, prob) in enumerate(zip(domain_names, probs)):
        s = '%s: %.1f%%'%(domain_name, 100.0 * prob)
        if i == i_max:
            s = '*' + s

        print(s)

def usage():
    print('Usage: python domain_classify_an_image.py <experiment_dir> <image>')

if __name__ == '__main__':
    domain_classify_an_image(*(sys.argv[1:]))
