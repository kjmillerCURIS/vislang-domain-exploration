import os
import sys
import pickle
from experiment_params.param_utils import get_params_key
from tablify_subspace_stats_dict import tablify_subspace_stats_dict_helper

#tablify_subspace_stats_dict_helper(my_dict, embedding_type, f):

def tablify_corrupted_cifar10_subspace_stats_dict(experiment_dir):
    params_key = get_params_key(experiment_dir)
    with open(os.path.join(experiment_dir, 'subspace_analysis.pkl'), 'rb') as f:
        stats_dict = pickle.load(f)

    table_dir = os.path.join(experiment_dir, 'subspace_tables-%s'%(params_key))
    os.makedirs(table_dir, exist_ok=True)
    for split_key in sorted(stats_dict.keys()):
        for seen_key in sorted(stats_dict[split_key].keys()):
            for embedding_type in sorted(stats_dict[split_key][seen_key].keys()):
                my_dict = stats_dict[split_key][seen_key][embedding_type]
                table_filename = os.path.join(table_dir, '%s-%s-%s-%s-subspace_table.csv'%(params_key, split_key, seen_key, embedding_type))
                f = open(table_filename, 'w')
                tablify_subspace_stats_dict_helper(my_dict, embedding_type, f, aug_name='domain', num_explained_variances_1d=10, num_explained_variances_2d=10)
                f.close()

def usage():
    print('Usage: python tablify_corrupted_cifar10_subspace_stats_dict.py <experiment_dir>')

if __name__ == '__main__':
    tablify_corrupted_cifar10_subspace_stats_dict(*(sys.argv[1:]))
