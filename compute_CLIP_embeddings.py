import os
import sys
import cv2
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
import torch
import clip
from non_image_data_utils import load_non_image_data
from general_aug_utils import generate_aug_dict

DEBUG = True
DEBUG_NUM_PTS = 10
SAVE_FREQ = 1000
CLIP_MODEL_TYPE = 'ViT-B/32'

#returns 'cuda' or 'cpu', depending on what's available
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def compute_CLIP_image_embeddings(image_paths, aug_dict, model, preprocess, device, embedding_dict, embedding_dict_filename):
    t = 0
    for image_path in tqdm(image_paths):
        numI = cv2.imread(image_path)
        image_base = os.path.basename(image_path)
        if image_base not in embedding_dict['image']:
            embedding_dict['image'][image_base] = {}

        for augID in tqdm(sorted(aug_dict.keys())):
            if augID in embedding_dict['image'][image_base]:
                print('found existing image embedding for ("%s", "%s"), skipping computation'%(image_base, augID))
                continue

            if DEBUG and t > DEBUG_NUM_PTS:
                return embedding_dict

            if t > 0 and t % SAVE_FREQ == 0:
                with open(embedding_dict_filename, 'wb') as f:
                    pickle.dump(embedding_dict, f)

            aug_fn = aug_dict[augID]['image_aug_fn']
            numIaug = aug_fn(numI)
            img = Image.fromarray(cv2.cvtColor(numIaug, cv2.COLOR_BGR2RGB))
            img = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(img).cpu()

            embedding = np.squeeze(embedding.numpy())
            embedding_dict['image'][image_base][augID] = embedding
            t += 1

    return embedding_dict

def compute_CLIP_text_embeddings(class2words_dict, aug_dict, model, device, embedding_dict, embedding_dict_filename):
    t = 0
    for classID in tqdm(sorted(class2words_dict.keys())):
        assert(len(set(class2words_dict[classID])) == len(class2words_dict[classID]))

    for augID in tqdm(sorted(aug_dict.keys())):
        assert(len(set(aug_dict[augID]['text_aug_templates'])) == len(aug_dict[augID]['text_aug_templates']))

    for classID in tqdm(sorted(class2words_dict.keys())):
        if classID not in embedding_dict['text']:
            embedding_dict['text'][classID] = {}

        for className in class2words_dict[classID]:
            if className not in embedding_dict['text'][classID]:
                embedding_dict['text'][classID][className] = {}

            for augID in tqdm(sorted(aug_dict.keys())):
                if augID not in embedding_dict['text'][classID][className]:
                    embedding_dict['text'][classID][className][augID] = {}

                for text_aug_template in aug_dict[augID]['text_aug_templates']:
                    if text_aug_template in embedding_dict['text'][classID][className][augID]:
                        print('found existing image embedding for ("%s", "%s", "%s", "%s"), skipping computation'%(classID, className, augID, text_aug_template))
                        continue

                    if DEBUG and t > DEBUG_NUM_PTS:
                        return embedding_dict

                    if t > 0 and t % SAVE_FREQ == 0:
                        with open(embedding_dict_filename, 'wb') as f:
                            pickle.dump(embedding_dict, f)

                    text_query = text_aug_template % className
                    text_query = clip.tokenize([text_query]).to(device)
                    with torch.no_grad():
                        embedding = model.encode_text(text_query).cpu()

                    embedding_dict['text'][classID][className][augID][text_aug_template] = embedding
                    t += 1

    return embedding_dict

def compute_CLIP_embeddings(base_dir, embedding_dict_filename):
    base_dir = os.path.abspath(os.path.expanduser(base_dir))

    device = get_device()
    embedding_dict = {'image' : {}, 'text' : {}}
    if os.path.exists(embedding_dict_filename):
        print('found existing embedding dict "%s", will populate anything that is missing there'%(embedding_dict_filename))
        with open(embedding_dict_filename, 'rb') as f:
            embedding_dict = pickle.load(f)

    embedding_dict['complete'] = False

    _, class2words_dict, class2filenames_dict = load_non_image_data(base_dir)
    image_paths = []
    for classID in tqdm(sorted(class2filenames_dict.keys())):
        image_paths.extend(class2filenames_dict[classID])

    aug_dict = generate_aug_dict()
    model, preprocess = clip.load(CLIP_MODEL_TYPE, device=device)

    embedding_dict = compute_CLIP_image_embeddings(image_paths, aug_dict, model, preprocess, device, embedding_dict, embedding_dict_filename)
    with open(embedding_dict_filename, 'wb') as f:
        pickle.dump(embedding_dict, f)

    embedding_dict = compute_CLIP_text_embeddings(class2words_dict, aug_dict, model, device, embedding_dict, embedding_dict_filename)
    if not DEBUG:
        embedding_dict['complete'] = True

    with open(embedding_dict_filename, 'wb') as f:
        pickle.dump(embedding_dict, f)

def usage():
    print('Usage: python compute_CLIP_embeddings.py <base_dir> <embedding_dict_filename>')

if __name__ == '__main__':
    compute_CLIP_embeddings(*(sys.argv[1:]))
