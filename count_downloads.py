import os
import sys
import glob
from tqdm import tqdm

def count_downloads(experiment_dir):
    total = 0
    domain_dirs = sorted(glob.glob(os.path.join(experiment_dir, 'laion_sample', '*')))
    for domain_dir in domain_dirs:
        sub_dirs = sorted(glob.glob(os.path.join(domain_dir, 'images', '*')))
        for sub_dir in sub_dirs:
            if not os.path.isdir(sub_dir):
                continue

            meow = len(glob.glob(os.path.join(sub_dir, '*.jpg')))
            print('"%s" ==> %d'%(sub_dir, meow))
            total += meow
            print('total = %d'%(total))

        print('')

if __name__ == '__main__':
    count_downloads(*(sys.argv[1:]))
