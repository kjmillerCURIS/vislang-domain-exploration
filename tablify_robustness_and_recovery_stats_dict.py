import os
import sys
import pickle
from plot_and_table_utils import ORDERED_AUG_IDS, format_percentage, format_cossim, format_p_value

def tablify_robustness_and_recovery_stats_dict(stats_dict_filename, stats_csv_filename):
    with open(stats_dict_filename, 'rb') as f:
        stats_dict = pickle.load(f)

    f = open(stats_csv_filename, 'w')

    #cosine similarity
    f.write('augID,mean_decrease,SD_decrease,p_decrease,mean_recovery,SD_recovery,p_recovery,mean_decrease_diffclass,mean_recovery_diffclass\n')
    for augID in ORDERED_AUG_IDS:
        items = [augID]
        for k in ['mean_decrease', 'SD_decrease', 'p_decrease', 'mean_recovery', 'SD_recovery', 'p_recovery', 'mean_decrease_diffclass', 'mean_recovery_diffclass']:
            format_fn = format_cossim
            if len(k) >= 2 and k[:2] == 'p_':
                format_fn = format_p_value

            items.append(format_fn(stats_dict['cossim']['primary'][augID][k]))

        f.write(','.join(items) + '\n')

    f.write('\n')
    f.write('avg_cossim_sameclass_unaug:,' + format_cossim(stats_dict['cossim']['secondary']['avg_cossim_sameclass_unaug']) + '\n')
    f.write('avg_cossim_diffclass_unaug:,' + format_cossim(stats_dict['cossim']['secondary']['avg_cossim_diffclass_unaug']) + '\n')

    #zero-shot accuarcy
    for top_k in sorted(stats_dict['zeroshot'].keys()):
        f.write('\n\n')
        f.write('augID,top%dacc_decrease_as_percentage,top%dacc_recovery_as_percentage\n'%(top_k, top_k))
        for augID in ORDERED_AUG_IDS:
            items = [augID]
            for k in ['acc_decrease_as_percentage', 'acc_recovery_as_percentage']:
                items.append(format_percentage(stats_dict['zeroshot'][top_k]['primary'][augID][k]))

            f.write(','.join(items) + '\n')

        f.write('\n')
        f.write('unaugmented_top%dacc_as_percentage:,'%(top_k) + format_percentage(stats_dict['zeroshot'][top_k]['secondary']['unaugmented_acc_as_percentage']) + '\n')

    f.close()

def usage():
    print('Usage: python tablify_robustness_and_recovery_stats_dict.py <stats_dict_filename> <stats_csv_filename>')

if __name__ == '__main__':
    tablify_robustness_and_recovery_stats_dict(*(sys.argv[1:]))
