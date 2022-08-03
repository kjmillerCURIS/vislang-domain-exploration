import os
import sys
import cv2
cv2.setNumThreads(1)
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
import torch
import clip
from non_image_data_utils import load_non_image_data
from general_aug_utils import generate_aug_dict

DEBUG = False
DEBUG_NUM_PTS = 640 #number of embeddings
SAVE_FREQ = 1000 #number of embeddings
CHUNK_SIZE = 10000 #number of embeddings
CLIP_MODEL_TYPE = 'ViT-B/32'
IMAGE_BATCH_SIZE = 18 #just keep all the augs in one batch for now

BAD_IMAGE_PATHS = ['/projectnb2/ivc-ml/nivek/vislang-domain-exploration-data/ILSVRC2012_val/n04589890/ILSVRC2012_val_00002258.JPEG',
                    '/projectnb2/ivc-ml/nivek/vislang-domain-exploration-data/ILSVRC2012_val/n04589890/ILSVRC2012_val_00012796.JPEG',
                    '/projectnb2/ivc-ml/nivek/vislang-domain-exploration-data/ILSVRC2012_val/n04589890/ILSVRC2012_val_00028315.JPEG']

#in case I don't trust qsub
def write_to_log_file(msg):
    f = open('meow.txt', 'a')
    f.write(msg + '\n')
    f.close()

#returns 'cuda' or 'cpu', depending on what's available
def get_device():
    if torch.cuda.is_available():
        print('I am on the GPU!')
        write_to_log_file('I am on the GPU!')
        return 'cuda'
    else:
        print('I am on the CPU!')
        write_to_log_file('I am on the CPU!')
        return 'cpu'

#if chunk_t < start_chunk_t then we just go through the motions, pass back an empty dict
def load_or_create_chunk(chunk_type, chunk_t, embedding_dict_filename_prefix, start_chunk_t):
    assert(chunk_type in ['image', 'text'])
    if chunk_t < start_chunk_t:
        return {chunk_type : {}}

    embedding_dict_filename = embedding_dict_filename_prefix + '-' + chunk_type + '_chunk_%09d.pkl'%(chunk_t)
    if os.path.exists(embedding_dict_filename):
        print('found existing chunk "%s", will fill it in as needed'%(embedding_dict_filename))
        with open(embedding_dict_filename, 'rb') as f:
            embedding_dict = pickle.load(f)

        embedding_dict['complete'] = False
        return embedding_dict
    else:
        return {chunk_type : {}, 'complete' : False}

#if chunk_t < start_chunk_t then we just go through the motions, do nothing
def save_chunk(embedding_dict, chunk_type, chunk_t, embedding_dict_filename_prefix, start_chunk_t, complete=False):
    assert(chunk_type in ['image', 'text'])
    if chunk_t < start_chunk_t:
        return

    embedding_dict['complete'] = complete
    embedding_dict_filename = embedding_dict_filename_prefix + '-' + chunk_type + '_chunk_%09d.pkl'%(chunk_t)
    with open(embedding_dict_filename, 'wb') as f:
        pickle.dump(embedding_dict, f)

def compute_CLIP_image_embeddings(image_paths, aug_dict, model, preprocess, device, embedding_dict_filename_prefix, start_chunk_t=0):
    t = 0
    chunk_t = 0
    embedding_dict = load_or_create_chunk('image', chunk_t, embedding_dict_filename_prefix, start_chunk_t)
    for image_path in tqdm(image_paths):
        if image_path in BAD_IMAGE_PATHS:
            continue

        image_base = os.path.basename(image_path)
        if image_base not in embedding_dict['image']:
            embedding_dict['image'][image_base] = {}

        if all([augID in embedding_dict['image'][image_base] for augID in sorted(aug_dict.keys())]):
            print('found all aug image embeddings for image "%s", skipping computation'%(image_base))
            t += len(aug_dict)
            continue

        if chunk_t >= start_chunk_t:
            print(image_path)
            numI = cv2.imread(image_path)
            print('meow')
            imgs_list = [[]]
            for augID in tqdm(sorted(aug_dict.keys())):
                aug_fn = aug_dict[augID]['image_aug_fn']
                numIaug = aug_fn(numI)
                img = Image.fromarray(cv2.cvtColor(numIaug, cv2.COLOR_BGR2RGB))
                if len(imgs_list[-1]) >= IMAGE_BATCH_SIZE:
                    imgs_list.append([])

                imgs_list[-1].append(img)

            embeddings_list = []
            for imgs in imgs_list:
                imgs_tnsr = torch.cat([preprocess(img).unsqueeze(0) for img in imgs]).to(device)
                with torch.no_grad():
                    embeddings = model.encode_image(imgs_tnsr).cpu().numpy()

                embeddings_list.append(embeddings)

            embeddings = np.concatenate(embeddings_list)
            for augID, embedding in zip(sorted(aug_dict.keys()), embeddings):
                embedding_dict['image'][image_base][augID] = embedding

        t += len(aug_dict)
        if DEBUG and t > DEBUG_NUM_PTS:
            save_chunk(embedding_dict, 'image', chunk_t, embedding_dict_filename_prefix, start_chunk_t)
            return

        if t > 0 and t % SAVE_FREQ < (t - len(aug_dict)) % SAVE_FREQ:
            save_chunk(embedding_dict, 'image', chunk_t, embedding_dict_filename_prefix, start_chunk_t)

        if t >= CHUNK_SIZE:
            save_chunk(embedding_dict, 'image', chunk_t, embedding_dict_filename_prefix, start_chunk_t, complete=True)
            t = 0
            chunk_t += 1
            embedding_dict = load_or_create_chunk('image', chunk_t, embedding_dict_filename_prefix, start_chunk_t)

    save_chunk(embedding_dict, 'image', chunk_t, embedding_dict_filename_prefix, start_chunk_t, complete=True)

def compute_CLIP_text_embeddings(class2words_dict, aug_dict, model, device, embedding_dict_filename_prefix, start_chunk_t=0):
    t = 0
    chunk_t = 0
    for classID in tqdm(sorted(class2words_dict.keys())):
        assert(len(set(class2words_dict[classID])) == len(class2words_dict[classID]))

    for augID in tqdm(sorted(aug_dict.keys())):
        assert(len(set(aug_dict[augID]['text_aug_templates'])) == len(aug_dict[augID]['text_aug_templates']))

    embedding_dict = load_or_create_chunk('text', chunk_t, embedding_dict_filename_prefix, start_chunk_t)
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
                        print('found existing text embedding for ("%s", "%s", "%s", "%s"), skipping computation'%(classID, className, augID, text_aug_template))
                        t += 1
                        continue

                    if DEBUG and t > DEBUG_NUM_PTS:
                        save_chunk(embedding_dict, 'text', chunk_t, embedding_dict_filename_prefix, start_chunk_t)
                        return

                    if t > 0 and t % SAVE_FREQ == 0:
                        save_chunk(embedding_dict, 'text', chunk_t, embedding_dict_filename_prefix, start_chunk_t)

                    if chunk_t >= start_chunk_t:
                        text_query = text_aug_template % className
                        text_query = clip.tokenize([text_query]).to(device)
                        with torch.no_grad():
                            embedding = model.encode_text(text_query).cpu()

                        embedding_dict['text'][classID][className][augID][text_aug_template] = embedding

                    t += 1
                    if t >= CHUNK_SIZE:
                        save_chunk(embedding_dict, 'text', chunk_t, embedding_dict_filename_prefix, start_chunk_t, complete=True)
                        t = 0
                        chunk_t += 1
                        embedding_dict = load_or_create_chunk('text', chunk_t, embedding_dict_filename_prefix, start_chunk_t)
                        if classID not in embedding_dict['text']:
                            embedding_dict['text'][classID] = {}
                        if className not in embedding_dict['text'][classID]:
                            embedding_dict['text'][classID][className] = {}
                        if augID not in embedding_dict['text'][classID][className]:
                            embedding_dict['text'][classID][className][augID] = {}

    save_chunk(embedding_dict, 'text', chunk_t, embedding_dict_filename_prefix, start_chunk_t, complete=True)

def compute_CLIP_embeddings(base_dir, embedding_dict_filename_prefix, image_start_chunk_t, text_start_chunk_t):
    base_dir = os.path.abspath(os.path.expanduser(base_dir))
    image_start_chunk_t = int(image_start_chunk_t)
    text_start_chunk_t = int(text_start_chunk_t)
    os.makedirs(os.path.dirname(embedding_dict_filename_prefix), exist_ok=True)

    device = get_device()

    _, class2words_dict, class2filenames_dict = load_non_image_data(base_dir)
    image_paths = []
    for classID in tqdm(sorted(class2filenames_dict.keys())):
        image_paths.extend(class2filenames_dict[classID])

    aug_dict = generate_aug_dict()
    model, preprocess = clip.load(CLIP_MODEL_TYPE, device=device)

#    compute_CLIP_image_embeddings(image_paths, aug_dict, model, preprocess, device, embedding_dict_filename_prefix, start_chunk_t=image_start_chunk_t)
    compute_CLIP_text_embeddings(class2words_dict, aug_dict, model, device, embedding_dict_filename_prefix, start_chunk_t=text_start_chunk_t)

def usage():
    print('Usage: python compute_CLIP_embeddings.py <base_dir> <embedding_dict_filename_prefix> <image_start_chunk_t> <text_start_chunk_t>')

if __name__ == '__main__':
    compute_CLIP_embeddings(*(sys.argv[1:]))
