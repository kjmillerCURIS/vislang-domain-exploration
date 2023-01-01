import os
import sys
import cv2
import glob
import numpy as np
import pickle
from tqdm import tqdm

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
NUM_CHANNELS = 3

#yeilds image_path, numI, my_class
def loading_generator(batch_filename, image_path_prefix):
    with open(batch_filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')

    for t, (image_data, my_class) in enumerate(zip(batch[b'data'], batch[b'labels'])):
        numI = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), dtype='uint8')
        for k in range(NUM_CHANNELS):
            numI[:,:,NUM_CHANNELS-k-1] = np.reshape(image_data[k * IMAGE_HEIGHT * IMAGE_WIDTH:(k+1) * IMAGE_HEIGHT * IMAGE_WIDTH], (IMAGE_HEIGHT, IMAGE_WIDTH))

        image_path = image_path_prefix + '_%07d_%d.png'%(t, my_class)
        yield image_path, numI, my_class

def extract_cifar10_images_train(compressed_cifar10_dir, cifar10_dir):
    dst_dir = os.path.join(cifar10_dir, 'train')
    image_dir = os.path.join(dst_dir, 'images')
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    class_dict = {}
    batch_filenames = sorted(glob.glob(os.path.join(compressed_cifar10_dir, 'data_batch_*')))
    for batch_filename in batch_filenames:
        image_path_prefix = os.path.join(image_dir, os.path.basename(batch_filename).split('_')[-1])
        my_genny = loading_generator(batch_filename, image_path_prefix)
        for image_path, numI, my_class in my_genny:
            cv2.imwrite(image_path, numI)
            class_dict[os.path.basename(image_path)] = {'class' : my_class}

    with open(os.path.join(dst_dir, 'class_dict.pkl'), 'wb') as f:
        pickle.dump(class_dict, f)

def extract_cifar10_images_test(compressed_cifar10_dir, cifar10_dir):
    dst_dir = os.path.join(cifar10_dir, 'test')
    image_dir = os.path.join(dst_dir, 'images')
    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    class_dict = {}
    batch_filename = os.path.join(compressed_cifar10_dir, 'test_batch')
    image_path_prefix = os.path.join(image_dir, 'test')
    my_genny = loading_generator(batch_filename, image_path_prefix)
    for image_path, numI, my_class in my_genny:
        cv2.imwrite(image_path, numI)
        class_dict[os.path.basename(image_path)] = {'class' : my_class}

    with open(os.path.join(dst_dir, 'class_dict.pkl'), 'wb') as f:
        pickle.dump(class_dict, f)

def extract_cifar10_images(compressed_cifar10_dir, cifar10_dir):
    extract_cifar10_images_train(compressed_cifar10_dir, cifar10_dir)
    extract_cifar10_images_test(compressed_cifar10_dir, cifar10_dir)

def usage():
    print('Usage: python extract_cifar10_images.py <compressed_cifar10_dir> <cifar10_dir>')

if __name__ == '__main__':
    extract_cifar10_images(*(sys.argv[1:]))
