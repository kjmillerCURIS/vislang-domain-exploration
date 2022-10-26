import os
import sys
import glob
from img2dataset import download
from compute_CLIP_embeddings import write_to_log_file

IMAGE_SIZE = 224
DEFAULT_PROCESS_COUNT = 4
ENCODE_FORMAT = 'jpg'
ENCODE_QUALITY = 95

def get_domain_dir(experiment_dir, domain_index):
    
    #special case
    if domain_index == -1:
        return os.path.join(experiment_dir, 'laion_sample', 'uniform_subset')

    candidates = sorted(glob.glob(os.path.join(experiment_dir, 'laion_sample', '%03d-*'%(domain_index))))
    assert(len(candidates) == 1)
    return candidates[0]

def download_images_one_domain(domain_dir, process_count=DEFAULT_PROCESS_COUNT):
    url_filename = os.path.join(domain_dir, 'image_urls.txt')
    output_dir = os.path.join(domain_dir, 'images')
    os.makedirs(output_dir, exist_ok=True)
    download(image_size=IMAGE_SIZE, processes_count=process_count, url_list=url_filename, output_folder=output_dir, output_format='files', input_format='txt', resize_mode='keep_ratio', encode_format=ENCODE_FORMAT, encode_quality=ENCODE_QUALITY)

def download_images_from_laion(experiment_dir, domain_indices, process_count=DEFAULT_PROCESS_COUNT):
    experiment_dir = os.path.abspath(os.path.expanduser(experiment_dir))
    domain_indices = [int(s) for s in domain_indices.split(',')]
    process_count = int(process_count)

    domain_dirs = [get_domain_dir(experiment_dir, domain_index) for domain_index in domain_indices]
    for domain_dir in domain_dirs:
        write_to_log_file('downloading to "%s"...'%(domain_dir))
        download_images_one_domain(domain_dir, process_count=process_count)
        write_to_log_file('done downlaoding to %s'%(domain_dir))

def usage():
    print('Usage: python download_images_from_laion.py <experiment_dir> <domain_indices> [<process_count>=%d]'%(DEFAULT_PROCESS_COUNT))

if __name__ == '__main__':
    download_images_from_laion(*(sys.argv[1:]))
