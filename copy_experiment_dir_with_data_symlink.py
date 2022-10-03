import os
import sys
import pickle
import shutil
from experiment_params.balance_params import grab_params
from experiment_params.param_utils import get_params_key

def copy_experiment_dir_with_data_symlink(src_experiment_dir, params_key, dst_experiment_dir):
    src_experiment_dir = os.path.abspath(os.path.expanduser(src_experiment_dir))
    dst_experiment_dir = os.path.abspath(os.path.expanduser(dst_experiment_dir))
    p = grab_params(params_key)
    src_params_key = get_params_key(src_experiment_dir)
    assert(src_params_key in p.data_compatible_params_keys)
    assert(not os.path.exists(dst_experiment_dir))
    os.makedirs(dst_experiment_dir, exist_ok=True)
    with open(os.path.join(dst_experiment_dir, 'params_key.pkl'), 'wb') as f:
        pickle.dump(params_key, f)

    shutil.copy(os.path.join(src_experiment_dir, 'train_domain_filter.pkl'), dst_experiment_dir)
    os.symlink(os.path.join(src_experiment_dir, 'laion_sample'), os.path.join(dst_experiment_dir, 'laion_sample'))

def usage():
    print('Usage: python copy_experiment_dir_with_data_symlink.py <src_experiment_dir> <params_key> <dst_experiment_dir>')

if __name__ == '__main__':
    copy_experiment_dir_with_data_symlink(*(sys.argv[1:]))
