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
from chunk_writer import ChunkWriter

DEBUG = False
DEBUG_NUM_PTS = 180 #number of embeddings
SAVE_FREQ = 1000 #number of embeddings
CHUNK_SIZE = 10008 #number of embeddings (made it multiple of IMAGE_BATCH_SIZE to minimize unnecessary file IO)
CLIP_MODEL_TYPE = 'ViT-B/32'
IMAGE_BATCH_SIZE = 18 #just keep all the augs in one batch for now

BAD_IMAGE_BASES = ['ILSVRC2012_val_00002258.JPEG',
                    'ILSVRC2012_val_00012796.JPEG',
                    'ILSVRC2012_val_00028315.JPEG']

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

def compute_CLIP_image_embeddings(image_paths, aug_dict, model, preprocess, device, embedding_dict_filename_prefix):
    debug_counter = 0
    
    #setup chunk-writer
    all_keys = []
    for image_path in image_paths:
        if os.path.basename(image_path) in BAD_IMAGE_BASES:
            continue

        for augID in sorted(aug_dict.keys()):
            all_keys.append((os.path.basename(image_path), augID))

    my_chunk_writer = ChunkWriter(CHUNK_SIZE, SAVE_FREQ, 'image', all_keys, embedding_dict_filename_prefix)

    #now the main iteration
    for image_path in tqdm(image_paths):
        if os.path.basename(image_path) in BAD_IMAGE_BASES:
            continue

        image_base = os.path.basename(image_path)
        all_augs_already_computed = True
        for augID in sorted(aug_dict.keys()):
            k = (image_base, augID)
            if not my_chunk_writer.contains(k):
                all_augs_already_computed = False
                break

        if all_augs_already_computed:
            print('found all aug image embeddings for image "%s", skipping computation'%(image_base))
            continue

        numI = cv2.imread(image_path)

        #split up augmented images into batches
        imgs_list = [[]]
        for augID in tqdm(sorted(aug_dict.keys())):
            aug_fn = aug_dict[augID]['image_aug_fn']
            numIaug = aug_fn(numI)
            img = Image.fromarray(cv2.cvtColor(numIaug, cv2.COLOR_BGR2RGB))
            if len(imgs_list[-1]) >= IMAGE_BATCH_SIZE:
                imgs_list.append([])

            imgs_list[-1].append(img)

        #now inference each batch
        embeddings_list = []
        for imgs in imgs_list:
            imgs_tnsr = torch.cat([preprocess(img).unsqueeze(0) for img in imgs]).to(device)
            with torch.no_grad():
                embeddings = model.encode_image(imgs_tnsr).cpu().numpy()

            embeddings_list.append(embeddings)

        #concatenate the outputs
        embeddings = np.concatenate(embeddings_list)
        assert(len(embeddings) == len(aug_dict))
        debug_counter += len(embeddings)

        #put each key-value pair into chunk-writer
        for augID, embedding in zip(sorted(aug_dict.keys()), embeddings):
            k = (image_base, augID)
            my_chunk_writer.insert(k, embedding)

        if DEBUG and debug_counter >= DEBUG_NUM_PTS:
            break

    my_chunk_writer.save()

def compute_CLIP_text_embeddings(class2words_dict, aug_dict, model, device, embedding_dict_filename_prefix):
    debug_counter = 0

    #setup chunk-writer
    all_keys = []
    for classID in sorted(class2words_dict.keys()):
        assert(len(set(class2words_dict[classID])) == len(class2words_dict[classID]))
        for className in class2words_dict[classID]:
            for augID in sorted(aug_dict.keys()):
                assert(len(set(aug_dict[augID]['text_aug_templates'])) == len(aug_dict[augID]['text_aug_templates']))
                for text_aug_template in aug_dict[augID]['text_aug_templates']:
                    k = (classID, className, augID, text_aug_template)
                    all_keys.append(k)

    my_chunk_writer = ChunkWriter(CHUNK_SIZE, SAVE_FREQ, 'text', all_keys, embedding_dict_filename_prefix)

    #main iteration
    for classID in tqdm(sorted(class2words_dict.keys())):
        for className in class2words_dict[classID]:
            for augID in tqdm(sorted(aug_dict.keys())):
                for text_aug_template in aug_dict[augID]['text_aug_templates']:
                    k = (classID, className, augID, text_aug_template)
                    if my_chunk_writer.contains(k):
                        print('found existing text embedding for "%s", skipping computation'%(str(k)))
                        continue

                    text_query = text_aug_template % className
                    text_query = clip.tokenize([text_query]).to(device)
                    with torch.no_grad():
                        embedding = np.squeeze(model.encode_text(text_query).cpu().numpy())

                    my_chunk_writer.insert(k, embedding)
                    debug_counter += 1
                    if DEBUG and debug_counter >= DEBUG_NUM_PTS:
                        break

    my_chunk_writer.save()

def compute_CLIP_embeddings(base_dir, embedding_dict_filename_prefix):
    base_dir = os.path.abspath(os.path.expanduser(base_dir))
    os.makedirs(os.path.dirname(embedding_dict_filename_prefix), exist_ok=True)

    device = get_device()

    _, class2words_dict, class2filenames_dict = load_non_image_data(base_dir)
    image_paths = []
    for classID in tqdm(sorted(class2filenames_dict.keys())):
        image_paths.extend(class2filenames_dict[classID])

    aug_dict = generate_aug_dict()
    model, preprocess = clip.load(CLIP_MODEL_TYPE, device=device)

    compute_CLIP_image_embeddings(image_paths, aug_dict, model, preprocess, device, embedding_dict_filename_prefix)
    compute_CLIP_text_embeddings(class2words_dict, aug_dict, model, device, embedding_dict_filename_prefix)

def usage():
    print('Usage: python compute_CLIP_embeddings.py <base_dir> <embedding_dict_filename_prefix>')

if __name__ == '__main__':
    compute_CLIP_embeddings(*(sys.argv[1:]))
