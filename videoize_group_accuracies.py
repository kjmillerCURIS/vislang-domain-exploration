import os
import sys
import cv2
import glob
import numpy as np
from tqdm import tqdm
from command_history_utils import write_to_history

def get_size(image_paths):
    w = 0
    h = None
    for image_path in image_paths:
        numI = cv2.imread(image_path)
        if h is not None:
            assert(numI.shape[0] == h)
        else:
            h = numI.shape[0]

        w += numI.shape[1]

    return (w, h)

def videoize_group_accuracies_one(experiment_dir, video_dir):
    video_filename = os.path.join(video_dir, 'group_accuracies-' + os.path.basename(experiment_dir).replace('experiment_', '') + '.avi')
    print(experiment_dir)
    avg_domains_images = sorted(glob.glob(os.path.join(experiment_dir, 'group_accuracies_animations', 'avg_domains', '*.png')))
    own_domain_images = sorted(glob.glob(os.path.join(experiment_dir, 'group_accuracies_animations', 'own_domain', '*.png')))
    assert(len(avg_domains_images) == len(own_domain_images))
    size = get_size([avg_domains_images[0], own_domain_images[0]])
    my_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'MJPG'), 2, size)
    for imageA, imageB in zip(avg_domains_images, own_domain_images):
        numIA = cv2.imread(imageA)
        numIB = cv2.imread(imageB)
        numIvis = np.hstack((numIA, numIB))
        my_writer.write(numIvis)

    my_writer.release()

def videoize_group_accuracies(experiment_dirs, video_dir):
    os.makedirs(video_dir, exist_ok=True)
    write_to_history(video_dir)

    if not isinstance(experiment_dirs, list):
        experiment_dirs = experiment_dirs.split(',')

    for experiment_dir in tqdm(experiment_dirs):
        videoize_group_accuracies_one(experiment_dir, video_dir)

def usage():
    print('Usage: python videoize_group_accuracies.py <experiment_dirs> <video_dir>')

if __name__ == '__main__':
    videoize_group_accuracies(*(sys.argv[1:]))
