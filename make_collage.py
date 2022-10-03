import os
import sys
import cv2
import glob
import numpy as np
import random
from tqdm import tqdm

NUM_ROWS = 5
COLLAGE_WIDTH = 1920
IMAGE_HEIGHT = 224
BUFFER_THICKNESS = 20
SAMPLE_SIZE = 100

def make_collage_one_domain(domain_dir):
    images = sorted(glob.glob(os.path.join(domain_dir, 'images', '*', '*.jpg')))
    images = random.sample(images, 100)
    rows = [[]]
    cur_x = 0
    for image in images:
        numI = cv2.imread(image)
        if numI.shape[0] != IMAGE_HEIGHT:
            w = int(round(numI.shape[1] / numI.shape[0] * IMAGE_HEIGHT))
            numI = cv2.resize(numI, (w, IMAGE_HEIGHT))
        if cur_x + numI.shape[1] > COLLAGE_WIDTH:
            rows[-1].append(255 * np.ones((IMAGE_HEIGHT, COLLAGE_WIDTH - cur_x, 3), dtype='uint8'))
            cur_x = 0
            if len(rows) >= NUM_ROWS:
                break
            else:
                rows.append([])

        rows[-1].append(numI)
        cur_x += numI.shape[1]
        if cur_x + BUFFER_THICKNESS > COLLAGE_WIDTH:
            rows[-1].append(255 * np.ones((IMAGE_HEIGHT, COLLAGE_WIDTH - cur_x, 3), dtype='uint8'))
            cur_x = 0
            if len(rows) >= NUM_ROWS:
                break
            else:
                rows.append([])

        cur_x += BUFFER_THICKNESS
        rows[-1].append(255 * np.ones((IMAGE_HEIGHT, BUFFER_THICKNESS, 3), dtype='uint8'))

    stuffs = []
    for row in rows:
        stuffs.append(np.hstack(row))
        stuffs.append(255 * np.ones((BUFFER_THICKNESS, COLLAGE_WIDTH, 3), dtype='uint8'))

    numIcollage = np.vstack(stuffs)

    return numIcollage

def make_collage(experiment_dir):
    experiment_dir = os.path.expanduser(experiment_dir)
    output_dir = os.path.join(experiment_dir, 'collage')
    os.makedirs(output_dir, exist_ok=True)
    domain_dirs = sorted(glob.glob(os.path.join(experiment_dir, 'laion_sample', '*')))
    for domain_dir in tqdm(domain_dirs):
        numIcollage = make_collage_one_domain(domain_dir)
        outname = os.path.join(output_dir, os.path.basename(domain_dir) + '.jpg')
        print(outname)
        cv2.imwrite(outname, numIcollage)

def usage():
    print('Usage: python make_collage.py <experiment_dir>')

if __name__ == '__main__':
    make_collage(*(sys.argv[1:]))
