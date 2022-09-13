import os
import sys
import copy
import random
from general_aug_utils import generate_aug_dict
from experiment_params.balance_params import grab_params

def check_splits(splits, domains):
    for split in splits:
        assert(sorted(set(split['train']).union(set(split['test']))) == sorted(domains))
        assert(len(set(split['train']).intersection(set(split['test']))) == 0)

#domains should already be shuffled if we were intending to shuffle it (e.g. for k_fold, but not for leave_one_out)
def make_splits(domains, num_folds):
    assert(len(domains) % num_folds == 0)
    splits = [{'train' : [], 'test' : []} for k in range(num_folds)]
    for i, domain in enumerate(domains):
        k = i % num_folds
        splits[k]['test'].append(domain)
        for j in range(num_folds):
            if j == k:
                continue

            splits[j]['train'].append(domain)

    check_splits(splits, domains)
    return splits

#params should be a params object with fields "domain_type", "domain_split_type", and optionally "domain_split_seed" and "domain_num_folds"
#will return a list of {'train' : list-of-domains, 'test' : list-of-domains} dicts
def sample_domains(params):
    p = params
    if p.domain_type == 'synthetic':
        sorted_domains = sorted(generate_aug_dict().keys())
    else:
        assert(False)

    if p.domain_split_type == 'all_all': #in this case we train and test on the entire set of domains. So 1 "split".
        return {'train' : copy.deepcopy(sorted_domains), 'test' : copy.deepcopy(sorted_domains)}
    elif p.domain_split_type in ['leave_one_out', 'k_fold']:
        domains = copy.deepcopy(sorted_domains)
        if p.domain_split_type == 'leave_one_out':
            num_folds = len(domains)
        else:
            assert(p.domain_split_type == 'k_fold')
            random.seed(p.domain_split_seed)
            random.shuffle(domains)
            num_folds = p.domain_num_folds

        return make_splits(domains, num_folds)
    else:
        assert(False)

if __name__ == '__main__':
    p = grab_params('BalanceTextHead2ToeSynthetic6FoldPureZeroShotParams')
    splits = sample_domains(p)
    print(splits)
    import pdb
    pdb.set_trace()
