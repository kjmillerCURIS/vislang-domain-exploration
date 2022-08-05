import os
import sys
import json
from non_image_data_utils import load_non_image_data

def jsonify_class_words_for_the_1k_classes():
    _, class2words_dict, __ = load_non_image_data(os.path.expanduser('~/data/vislang-domain-exploration-data/ILSVRC2012_val'))
    s = json.dumps(class2words_dict)
    s = s.replace('], ', '],\n')
    with open('names_for_the_1k_classes.json', 'w') as f:
        f.write(s)

if __name__ == '__main__':
    jsonify_class_words_for_the_1k_classes()
