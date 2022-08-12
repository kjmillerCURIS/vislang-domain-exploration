import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import clip
from CLIP_paper_official_zeroshot_text_utils import OFFICIAL_CLIP_SORTED_CLASS_NAMES, OFFICIAL_CLIP_TEXT_TEMPLATES
from compute_CLIP_embeddings import get_device

CLIP_MODEL_TYPE = 'ViT-B/32'

def compute_CLIP_paper_zeroshot_classifier(classifier_npy_filename):
    assert(len(OFFICIAL_CLIP_SORTED_CLASS_NAMES) == 1000)
    assert(len(OFFICIAL_CLIP_TEXT_TEMPLATES) == 80)

    device = get_device()
    model, preprocess = clip.load(CLIP_MODEL_TYPE, device=device)

    classifier = []
    for className in tqdm(OFFICIAL_CLIP_SORTED_CLASS_NAMES):
        text_embeddings = []
        for text_template in tqdm(OFFICIAL_CLIP_TEXT_TEMPLATES):
            text_query = text_template.format(className)
            text_query = clip.tokenize([text_query]).to(device)
            with torch.no_grad():
                embedding = np.squeeze(model.encode_text(text_query).cpu().numpy()).astype('float64')

            embedding = embedding / np.linalg.norm(embedding)
            text_embeddings.append(embedding)

        text_embedding = np.mean(text_embeddings, axis=0)
        text_embedding = text_embedding / np.linalg.norm(text_embedding)
        classifier.append(text_embedding)

    classifier = np.array(classifier)

    np.save(classifier_npy_filename, classifier)

def usage():
    print('Usage: python compute_CLIP_paper_zeroshot_classifier.py <classifier_npy_filename>')

if __name__ == '__main__':
    compute_CLIP_paper_zeroshot_classifier(*(sys.argv[1:]))
