import os
import sys
import pickle

def tablify_robustness_and_recovery_stats_dict(stats_dict_filename, stats_csv_filename):
    with open(stats_dict_filename, 'rb') as f:
        stats_dict = pickle.load(f)

    f = open(stats_csv_filename, 'w')

    #cosine similarity
    f.write('augID,mean_decrease,SD_decrease,p_decrease,mean_recovery,SD_recovery,p_recovery,mean_decrease_diffclass,mean_recovery_diffclass\n')
    for augID in sorted(stats_dict['cossim']['primary'].keys()):
        items = [augID]
        for k in ['mean_decrease', 'SD_decrease', 'p_decrease', 'mean_recovery', 'SD_recovery', 'p_recovery', 'mean_decrease_diffclass', 'mean_recovery_diffclass']:
            items.append(str(stats_dict['cossim']['primary'][augID][k]))

        f.write(','.join(items) + '\n')

    f.write('\n')
    f.write('avg_cossim_sameclass_unaug:,' + str(stats_dict['cossim']['secondary']['avg_cossim_sameclass_unaug']) + '\n')
    f.write('avg_cossim_diffclass_unaug:,' + str(stats_dict['cossim']['secondary']['avg_cossim_diffclass_unaug']) + '\n')

    #zero-shot accuarcy
    for top_k in sorted(stats_dict['zeroshot'].keys()):
        f.write('\n\n')
        f.write('augID,top%dacc_decrease_as_percentage,top%dacc_recovery_as_percentage\n'%(top_k, top_k))
        for augID in sorted(stats_dict['zeroshot'][top_k]['primary'].keys()):
            items = [augID]
            for k in ['acc_decrease_as_percentage', 'acc_recovery_as_percentage']:
                items.append(str(stats_dict['zeroshot'][top_k]['primary'][augID][k]) + '%')

            f.write(','.join(items) + '\n')

        f.write('\n')
        f.write('unaugmented_top%dacc_as_percentage:,'%(top_k) + str(stats_dict['zeroshot'][top_k]['secondary']['unaugmented_acc_as_percentage']) + '%\n')

    f.close()

def usage():
    print('Usage: python tablify_robustness_and_recovery_stats_dict.py <stats_dict_filename> <stats_csv_filename>')

if __name__ == '__main__':
    tablify_robustness_and_recovery_stats_dict(*(sys.argv[1:]))
