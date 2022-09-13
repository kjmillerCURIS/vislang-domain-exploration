import os
import sys
import pickle
from plot_and_table_utils import ORDERED_AUG_IDS, format_percentage

def tablify_linearprobe_stats_dict(stats_dict_filename, stats_csv_filename):
    with open(stats_dict_filename, 'rb') as f:
         stats_dict = pickle.load(f)

    f = open(stats_csv_filename, 'w')
    f.write('augID,top1acc_decrease_as_percentage\n')
    for augID in ORDERED_AUG_IDS:
        f.write(augID + ',' + format_percentage(stats_dict['noop'] - stats_dict[augID]) + '\n')

    f.write('\n')
    f.write('unaugmented_top1acc_as_percentage,' + format_percentage(stats_dict['noop']) + '\n')

def usage():
    print('Usage: python tablify_linearprobe_stats_dict.py <stats_dict_filename> <stats_csv_filename>')

if __name__ == '__main__':
    tablify_linearprobe_stats_dict(*(sys.argv[1:]))
