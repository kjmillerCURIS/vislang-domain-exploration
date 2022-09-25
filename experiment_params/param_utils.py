import os
import sys
import pickle

def validate_params_key(experiment_dir, params_key):
    params_key_filename = os.path.join(experiment_dir, 'params_key.pkl')
    if not os.path.exists(params_key_filename):
        with open(params_key_filename, 'wb') as f:
            pickle.dump(params_key, f)

        print('"%s" did not exist, so we made one and populated it with "%s"'%(params_key_filename, params_key))
    else:
        with open(params_key_filename, 'rb') as f:
            existing_params_key = pickle.load(f)

        assert(params_key == existing_params_key)

def get_params_key(experiment_dir):
    params_key_filename = os.path.join(experiment_dir, 'params_key.pkl')
    with open(params_key_filename, 'rb') as f:
        params_key = pickle.load(f)

    return params_key
