import os
import sys
import numpy as np
import pickle
from plot_and_table_utils import NUM_EXPLAINED_VARIANCES_1D, NUM_EXPLAINED_VARIANCES_2D, format_spread, format_percentage, format_cossim, format_angle

def tablify_subspace_stats_dict(stats_dict_filename, stats_csv_filename):
    with open(stats_dict_filename, 'rb') as f:
        stats_dict = pickle.load(f)

    f = open(stats_csv_filename, 'w')
    for norm_str in ['normalized', 'unnormalized']:
        for embedding_type in ['image', 'text']:
            f.write(norm_str + ' ' + embedding_type + '\n')
            my_dict = stats_dict[norm_str][embedding_type]
            tablify_subspace_stats_dict_helper(my_dict, embedding_type, f)
            f.write('\n\n')

    f.close()

def tablify_subspace_stats_dict_helper(my_dict, embedding_type, f, aug_name='aug', num_explained_variances_1d=NUM_EXPLAINED_VARIANCES_1D, num_explained_variances_2d=NUM_EXPLAINED_VARIANCES_2D):

    #raw spreads
    itemsA = ['spread(additive_model_residuals)', 'spread(all_pairs)', 'spread(class_only)', 'spread(%s_only)'%(aug_name)]
    itemsB = [format_spread(my_dict[k]['spread']) for k in ['direction_residual_PCA', 'total_PCA', 'class_center_PCA', 'aug_center_PCA']]
    if embedding_type == 'image':
        itemsA.append('spread(image_deviations_from_means)')
        itemsB.append(format_spread(my_dict['deviation_PCA']['spread']))

    f.write(','.join(itemsA) + '\n')
    f.write(','.join(itemsB) + '\n')
    f.write('\n')

    #proportional spreads
    itemsA = ['', 'spread(additive_model_residuals) / spread(all_pairs) (%%)', 'spread(additive_model_residuals) / spread(class_only) (%%)', 'spread(additive_model_residuals) / spread(%s_only) (%%)'%(aug_name)]
    itemsB = [''] + [format_percentage(100.0 * my_dict['direction_residual_PCA']['spread'] / my_dict[k]['spread']) for k in ['total_PCA', 'class_center_PCA', 'aug_center_PCA']]
    if embedding_type == 'image':
        itemsA.append('spread(additive_model_residuals) / spread(image_deviations_from_means) (%)')
        itemsB.append(format_percentage(100.0 * my_dict['direction_residual_PCA']['spread'] / my_dict['deviation_PCA']['spread']))

    f.write(','.join(itemsA) + '\n')
    f.write(','.join(itemsB) + '\n')
    f.write('\n')

    #PCAs of decompositions - explained variance ranking
    f.write('explained variance percentages of class principal components\n')
    explained_variances_class = np.square(my_dict['class_comp_PCA']['explained_SDs'])
    explained_variance_percentages_class = 100.0 * explained_variances_class / np.sum(explained_variances_class)
    f.write(','.join([str(i) for i in range(1, num_explained_variances_1d + 1)]) + '\n')
    f.write(','.join([format_percentage(v) for v in explained_variance_percentages_class[:num_explained_variances_1d]]) + '\n')
    f.write('\n')
    f.write('explained variance percentages of %s principal components\n'%(aug_name))
    explained_variances_aug = np.square(my_dict['aug_comp_PCA']['explained_SDs'])
    explained_variance_percentages_aug = 100.0 * explained_variances_aug / np.sum(explained_variances_aug)
    f.write(','.join([str(i) for i in range(1, num_explained_variances_1d + 1)]) + '\n')
    f.write(','.join([format_percentage(v) for v in explained_variance_percentages_aug[:num_explained_variances_1d]]) + '\n')
    f.write('\n')

    #now get cossims between the principal components of the 2 decompositions
    for r in range(2):
        if r == 0:
            f.write('cosine similarity table\n')
        else:
            f.write('cosine angle (degrees) table\n')

        f.write(',' * (num_explained_variances_2d // 2) + 'explained variance %s\n'%(aug_name))
        f.write(',,' + ','.join([format_percentage(v) for v in explained_variance_percentages_aug[:num_explained_variances_2d]]) + '\n')
        for i, (row, v_class) in enumerate(zip(my_dict['class_aug_comp_PCA_cossims'][:num_explained_variances_2d,:], explained_variance_percentages_class[:num_explained_variances_2d:])):
            if i == num_explained_variances_2d // 2:
                items = ['explained variance class']
            else:
                items = ['']

            items.append(format_percentage(v_class))
            if r == 0:
                items.extend([format_cossim(np.fabs(z)) for z in row[:num_explained_variances_2d]])
            else:
                items.extend([format_angle(180.0 / np.pi * np.arccos(np.fabs(z))) for z in row[:num_explained_variances_2d]])

            f.write(','.join(items) + '\n')

        f.write('\n')

def usage():
    print('Usage: python tablify_subspace_stats_dict.py <stats_dict_filename> <stats_csv_filename>')

if __name__ == '__main__':
    tablify_subspace_stats_dict(*(sys.argv[1:]))
