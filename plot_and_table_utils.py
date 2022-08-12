import os
import sys

ORDERED_AUG_IDS = [
                    'make_background_white',
                    'make_background_black',
                    'make_background_blue',
                    'sketchify',
                    'posterize',
                    'low_res',
                    'blur',
                    'foggy',
                    'grayscale',
                    'sepia',
                    'dim',
                    'brighten',
                    'closeup',
                    'fisheye',
                    'upside_down',
                    'sideways',
                    'tilt'
                  ]

FACECOLOR = '#eaeaf2'
NUM_EXPLAINED_VARIANCES_1D = 30
NUM_EXPLAINED_VARIANCES_2D = 10

#v should already be multiplied by 100
def format_percentage(v : float):
    return '%.1f'%(v) + '%'

def format_cossim(v : float):
    return '%.5f'%(v)

def format_spread(v : float):
    return '%.2f'%(v)

#v should already be in degrees
def format_angle(v : float):
    return '%.1f'%(v)

def format_p_value(v : float):
    if v < 0.001:
        return '<0.001'

    return '%.3f'%(v)
