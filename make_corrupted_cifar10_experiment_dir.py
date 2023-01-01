import os
import sys
import pickle

def make_corrupted_cifar10_experiment_dir(params_key, train_type, experiment_dir):
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'params_key.pkl'), 'wb') as f:
        pickle.dump(params_key, f)

    with open(os.path.join(experiment_dir, 'train_type.pkl'), 'wb') as f:
        pickle.dump(train_type, f)

def usage():
    print('Usage: python make_corrupted_cifar10_experiment_dir.py <params_key> <train_type> <experiment_dir>')

if __name__ == '__main__':
    make_corrupted_cifar10_experiment_dir(*(sys.argv[1:]))
