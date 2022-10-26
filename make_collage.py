import os
import sys
import cv2
cv2.setNumThreads(1)
import glob
import numpy as np
import pickle
import random
from tqdm import tqdm

#common
SAMPLE_SIZE = 100

#image stuff
NUM_ROWS = 8
COLLAGE_WIDTH = 1600
IMAGE_HEIGHT = 224
BUFFER_THICKNESS = 20

#caption stuff
FONT_SIZE = 10
FONT_THICKNESS = 12
TEXT_BAR_HEIGHT = 150
TEXT_BAR_BUFFER_PROP = 0.1
TEXT_BAR_WIDTH_PER_CHAR = 150
MIN_TEXT_BAR_WIDTH_CHARS = 50
TEXT_TRIM_BUFFER = 100
TEXT_RESIZE_FACTOR = 5

def make_text_bar(caption):
    h = TEXT_BAR_HEIGHT
    buffer = int(round(TEXT_BAR_BUFFER_PROP * h))
    w = buffer + max(len(caption), MIN_TEXT_BAR_WIDTH_CHARS) * TEXT_BAR_WIDTH_PER_CHAR
    numItext = 255 * np.ones((h, w, 3), dtype='uint8')
    origin = (buffer, h - buffer)
    cv2.putText(numItext, caption, origin, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, (0,0,0), thickness=FONT_THICKNESS)
    return numItext

def merge_text_bars(numItext_list):
    max_width = max([numItext.shape[1] for numItext in numItext_list])
    numIrow_list = []
    for numItext in numItext_list:
        if numItext.shape[1] < max_width:
            numIrow = np.hstack((numItext, 255 * np.ones((TEXT_BAR_HEIGHT, max_width - numItext.shape[1], 3), dtype='uint8')))
            numIrow_list.append(numIrow)
        else:
            assert(numItext.shape[1] == max_width)
            numIrow_list.append(numItext)

    return np.vstack(numIrow_list)

def trim_and_resize_text_part(numIcollage_text):
    column_mins = np.squeeze(np.amin(numIcollage_text[:,:,1], axis=0)) #assume green channel will be 255 for background and <255 for text
    text_indices = np.nonzero(column_mins < 255)[0]
    x_end = min(np.amax(text_indices) + TEXT_TRIM_BUFFER, numIcollage_text.shape[1])
    numIcollage_text = numIcollage_text[:,:x_end,:]
    numIcollage_text = cv2.resize(numIcollage_text, None, fx=1/TEXT_RESIZE_FACTOR, fy=1/TEXT_RESIZE_FACTOR)
    return numIcollage_text

#returns an npy array
def make_text_part(domain_dir, image_indices_used):
    with open(os.path.join(domain_dir, 'image_bases.pkl'), 'rb') as f:
        image_bases = pickle.load(f)

    with open(os.path.join(domain_dir, 'image_base_to_caption.pkl'), 'rb') as f:
        caption_dict = pickle.load(f)

    captions = [caption_dict[image_bases[i]] for i in image_indices_used]
    numItext_list = [make_text_bar(caption) for caption in captions]
    numIcollage_text = merge_text_bars(numItext_list)
    numIcollage_text = trim_and_resize_text_part(numIcollage_text)
    return numIcollage_text

def make_collage_one_domain(domain_dir):
    images = sorted(glob.glob(os.path.join(domain_dir, 'images', '*', '*.jpg')))
    images = random.sample(images, SAMPLE_SIZE)
    rows = [[]]
    cur_x = 0
    image_indices_used = []
    for image in images:
        numI = cv2.imread(image)
        if numI.shape[0] != IMAGE_HEIGHT:
            w = int(round(numI.shape[1] / numI.shape[0] * IMAGE_HEIGHT))
            numI = cv2.resize(numI, (w, IMAGE_HEIGHT))

        if numI.shape[1] >= COLLAGE_WIDTH:
            print('!!! had to throw away a too-wide image !!!')
            continue

        if cur_x + numI.shape[1] > COLLAGE_WIDTH:
            rows[-1].append(255 * np.ones((IMAGE_HEIGHT, COLLAGE_WIDTH - cur_x, 3), dtype='uint8'))
            cur_x = 0
            if len(rows) >= NUM_ROWS:
                break
            else:
                rows.append([])

        rows[-1].append(numI)
        image_indices_used.append(int(os.path.splitext(os.path.basename(image))[0]))
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
    numIcollage_text = make_text_part(domain_dir, image_indices_used)
    assert(numIcollage_text.shape[0] > 0)
    assert(numIcollage_text.shape[1] > 0)
    assert(numIcollage_text.shape[2] == 3)

    return numIcollage, numIcollage_text

def make_collage(experiment_dir):
    experiment_dir = os.path.abspath(os.path.expanduser(experiment_dir))
    output_dir = os.path.join(experiment_dir, 'collage')
    os.makedirs(output_dir, exist_ok=True)
    domain_dirs = sorted(glob.glob(os.path.join(experiment_dir, 'laion_sample', '*')))
    for domain_dir in tqdm(domain_dirs):
        numIcollage, numIcollage_text = make_collage_one_domain(domain_dir)
        outname = os.path.join(output_dir, os.path.basename(domain_dir) + '.jpg')
        outname_text = os.path.join(output_dir, os.path.basename(domain_dir) + '-text.jpg')
        print((outname, outname_text))
        cv2.imwrite(outname, numIcollage)
        cv2.imwrite(outname_text, numIcollage_text)
        assert(os.path.exists(outname_text))

def usage():
    print('Usage: python make_collage.py <experiment_dir>')

if __name__ == '__main__':
    make_collage(*(sys.argv[1:]))
