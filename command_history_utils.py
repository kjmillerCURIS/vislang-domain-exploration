import os
import sys
import pickle
import time

def write_to_history(my_dir):
    my_time = time.time()
    history_filename = os.path.join(my_dir, 'history.pkl')
    if not os.path.exists(history_filename):
        history = []
    else:
        with open(history_filename, 'rb') as f:
            history = pickle.load(f)

    history.append((my_time, sys.argv))
    with open(history_filename, 'wb') as f:
        pickle.dump(history, f)
