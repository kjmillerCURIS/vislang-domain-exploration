import os
import sys

class BalanceTextHead2ToeSynthetic6FoldPureZeroShotParams:
    def __init__(self):
        self.domain_type = 'synthetic'
        self.domain_split_type = 'k_fold'
        self.domain_split_seed = 0
        self.domain_num_folds = 6

def grab_params(params_key):
    return eval(params_key + '()')
