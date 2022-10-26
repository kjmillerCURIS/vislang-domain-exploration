import os
import sys
import glob
from tqdm import tqdm

def remove_jsons(experiment_dir):
    experiment_dir = os.path.abspath(os.path.expanduser(experiment_dir))
    json_paths = sorted(glob.glob(os.path.join(experiment_dir, 'laion_sample', '*', 'images', '*', '*.json')))
    for json_path in tqdm(json_paths):
        os.remove(json_path)

def usage():
    print('Usage: python remove_jsons.py <experiment_dir>')

if __name__ == '__main__':
    remove_jsons(*(sys.argv[1:]))
