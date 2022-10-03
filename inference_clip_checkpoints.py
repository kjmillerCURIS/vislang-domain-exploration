import os
import sys
import clip
import pickle
import torch
from tqdm import tqdm
from experiment_params.balance_params import grab_params
from experiment_params.param_utils import get_params_key
from clip_training_utils import grab_clip_backbones
from compute_CLIP_embeddings import compute_CLIP_image_embeddings, compute_CLIP_text_embeddings
from non_image_data_utils import load_non_image_data
from general_aug_utils import generate_aug_dict

#compute_CLIP_image_embeddings(image_paths, aug_dict, models, preprocess, device, embedding_dict_filename_prefixes, num_parts=1, part_start_index=0)
#compute_CLIP_text_embeddings(class2words_dict, aug_dict, models, device, embedding_dict_filename_prefixes, num_parts=1, part_start_index=0)

CLIP_MODEL_TYPE = 'ViT-B/32'

def set_cv_threads():
    import cv2
    cv2.setNumThreads(2)

def get_preprocess(clip_model_type):
    _, preprocess = clip.load(CLIP_MODEL_TYPE, device='cuda')
    return preprocess

def inference_clip_checkpoints(experiment_dir, checkpoint_suffixes, val_base_dir):
    experiment_dir = os.path.abspath(os.path.expanduser(experiment_dir))
    checkpoint_suffixes = checkpoint_suffixes.split(',')
    val_base_dir = os.path.abspath(os.path.expanduser(val_base_dir))

    #stuff
    set_cv_threads()

    #image preprocessing stuff
    clip_model_type = grab_params(get_params_key(experiment_dir)).clip_model_type
    preprocess = get_preprocess(clip_model_type)

    #val data setup stuff
    #get image_paths, aug_dict, class2words_dict
    _, class2words_dict, class2filenames_dict = load_non_image_data(val_base_dir)
    image_paths = []
    for classID in tqdm(sorted(class2filenames_dict.keys())):
        image_paths.extend(class2filenames_dict[classID])

    aug_dict = generate_aug_dict()

    #output setup stuff
    embedding_base_dir = os.path.join(experiment_dir, 'val_embeddings')
    os.makedirs(embedding_base_dir, exist_ok=True)

    #go through the checkpoint suffixes
    image_models = []
    text_models = []
    embedding_dict_filename_prefixes = []
    for checkpoint_suffix in tqdm(checkpoint_suffixes):

        #load a model
        image_backbone, text_backbone, _ = grab_clip_backbones(clip_model_type)
        checkpoint_filename = os.path.join(experiment_dir, 'clip_finetuning_checkpoints', 'clip_finetuning_checkpoint-' + checkpoint_suffix + '.pth')
        checkpoint = torch.load(checkpoint_filename)
        image_backbone.load_state_dict(checkpoint['model_state_dict']['image'])
        text_backbone.load_state_dict(checkpoint['model_state_dict']['text'])
        image_backbone.eval()
        text_backbone.eval()
        image_models.append(image_backbone)
        text_models.append(text_backbone)

        #setup output embedding dir and make embedding prefix
        #also, save pickle with info for the x-axis of the plot
        embedding_dir = os.path.join(embedding_base_dir, checkpoint_suffix)
        os.makedirs(embedding_dir, exist_ok=True)
        embedding_dict_filename_prefix = os.path.join(embedding_dir, checkpoint_suffix)
        embedding_dict_filename_prefixes.append(embedding_dict_filename_prefix)
        with open(os.path.join(embedding_dir, 'epoch_info_dict.pkl'), 'wb') as f:
            pickle.dump({'epoch' : checkpoint['epoch'], 'step_within_epoch' : checkpoint['step_within_epoch']}, f)

    #inference!
    compute_CLIP_image_embeddings(image_paths, aug_dict, image_models, preprocess, 'cuda', embedding_dict_filename_prefixes, 1, 0, has_encode_fn=False)
    compute_CLIP_text_embeddings(class2words_dict, aug_dict, text_models, 'cuda', embedding_dict_filename_prefixes, 1, 0, has_encode_fn=False)

def usage():
    print('Usage: python inference_clip_checkpoints.py <experiment_dir> <checkpoint_suffixes> <val_base_dir>')

if __name__ == '__main__':
    inference_clip_checkpoints(*(sys.argv[1:]))
