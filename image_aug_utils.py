import os
import sys
import copy
import cv2
import glob
import numpy as np
import random
from tqdm import tqdm
import albumentations

SEG_BBOX_FACTOR = 0.2 #how much to shrink the image frame to get the bbox
SEG_CONTRACTION_ALPHA = 0.25 #how much to shrink the bbox to get the definitely-foreground part
SEG_GC_DOWNSCALE_FACTOR = 8
SEG_GC_NUM_ITERS = 3
SEG_MEDIAN_KSIZE = 21
GAUSSIAN_BLUR_KSIZE = 15
DIMMING_FACTOR = 0.2
BRIGHTENING_FACTOR = 1.8
CLOSEUP_FACTOR = 2.0
FISHEYE_KS = [3.2, 3.2, 3.2]
FISHEYE_SCALE_COMPENSATION_FACTOR = 1.4
POSTERIZATION_BITS = 2
'''
SKETCH_BLUR_KSIZE_A = 7 #for the Otsu part
SKETCH_BLUR_KSIZE_B = 3 #for the actual Canny part
#SKETCH_EDGE_DENSITY_MULTIPLIER = 1.4
SKETCH_EDGE_DENSITY_MULTIPLIER = 3.5
SKETCH_CANNY_HIGH_THRESHOLDS = [25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0]
SKETCH_CANNY_RATIO = 0.5
'''
SKETCH_BLUR_KSIZE = 3
SKETCH_LAPLACIAN_KSIZE = 3
TILT_MIN_ANGLE = 10.0
TILT_MAX_ANGLE = 20.0
LOW_RES_FACTOR = 0.25
FOG_COEF = 0.9
FOG_ALPHA = 0.08

#FIXME: Find a better way to document which functions are augmentations. Like maybe a decorator?
#I know I could do a class for each augmentation, but that seems like too much gruntwork, there's gotta be a better way

'''
functions starting with 'aug_' are augmentations
they take the form f(numI) ==> numI
where numI is an npy array, BGR ordering, 0-255, uint8
they do not modify numI, but instead return an augmented copy
'''

#this is a helper function, not an augmentation
#numI should be npy array
#color should be in BGR order, 0-255, e.g. (0,0,0), (255,255,255), (255,0,0)
#will return a npy bool array which is the foreground mask
#this is non-deterministic (sorry) and not very accurate (extra sorry)
def cheap_segment(numI):
    bbox_factor = SEG_BBOX_FACTOR
    contraction_alpha = SEG_CONTRACTION_ALPHA
    gc_downscale_factor = SEG_GC_DOWNSCALE_FACTOR
    gc_num_iters = SEG_GC_NUM_ITERS
    median_ksize = SEG_MEDIAN_KSIZE

    #make a central bbox
    w = int(round((1 - bbox_factor) * numI.shape[1]))
    h = int(round((1 - bbox_factor) * numI.shape[0]))
    x = (numI.shape[1] - w) // 2
    y = (numI.shape[0] - h) // 2

    #anything outside the bbox is definite background
    #anything inside a contracted version of the definite foreground
    #anything else is probable foreground
    mask = np.zeros(numI.shape[:2],np.uint8)
    mask[:,:] = cv2.GC_BGD
    mask[x:x+w, y:y+h] = cv2.GC_PR_FGD
    x_start = int(x + contraction_alpha * w)
    x_end = int(x + (1 - contraction_alpha) * w)
    y_start = int(y + contraction_alpha * h)
    y_end = int(y + (1 - contraction_alpha) * h)
    mask[y_start:y_end, x_start:x_end] = cv2.GC_FGD

    #do grabcut
    numItiny = cv2.resize(numI, None, fx=1.0/gc_downscale_factor, fy=1.0/gc_downscale_factor)
    mask_tiny = cv2.resize(mask, None, fx=1.0/gc_downscale_factor, fy=1.0/gc_downscale_factor, interpolation=cv2.INTER_NEAREST)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(numItiny,mask_tiny,None,bgdModel,fgdModel,gc_num_iters,cv2.GC_INIT_WITH_MASK)
    mask = cv2.resize(mask_tiny, (numI.shape[1], numI.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = cv2.medianBlur(mask, median_ksize)

    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)

#this is an augmentation
#just returns a copy of the image
def aug_noop(numI):
    return copy.deepcopy(numI)

#this is an augmentation
#uses grabcut with fixed seed mask to segment out foreground (or at least try)
#turns background BGR=(255,255,255)
def aug_make_background_white(numI):
    numI = copy.deepcopy(numI)
    mask = cheap_segment(numI)
    numI[~mask,:] = (255,255,255)
    return numI

#this is an augmentation
#uses grabcut with fixed seed mask to segment out foreground (or at least try)
#turns background BGR=(0,0,0)
def aug_make_background_black(numI):
    numI = copy.deepcopy(numI)
    mask = cheap_segment(numI)
    numI[~mask,:] = (0,0,0)
    return numI

#this is an augmentation
#uses grabcut with fixed seed mask to segment out foreground (or at least try)
#turns background BGR=(255,0,0)
def aug_make_background_blue(numI):
    numI = copy.deepcopy(numI)
    mask = cheap_segment(numI)
    numI[~mask,:] = (255,0,0)
    return numI

#this is an augmentation
#do a Laplacian filter, then subtract the absolute value from 255
def aug_sketchify(numI):
    ksize = SKETCH_BLUR_KSIZE
    laplacian_ksize = SKETCH_LAPLACIAN_KSIZE
    numIgray = cv2.cvtColor(numI, cv2.COLOR_BGR2GRAY)
    numIblur = cv2.GaussianBlur(numIgray, (ksize, ksize), 0)
    numIlaplacian = cv2.Laplacian(numIblur, cv2.CV_16S, ksize=laplacian_ksize)
    numIlaplacian = cv2.convertScaleAbs(numIlaplacian) #this apparently takes the absolute value AND scales and converts it to uint8

    #subtract from 255 to get a black-on-white drawing
    numIaug = 255 * np.ones_like(numI)
    numIaug = numIaug - numIlaplacian[:,:,np.newaxis]

    return numIaug

'''
#this is an augmentation
#use Otsu to get a hint on what the edge-density should be like
#then adjust Canny threshold to get closest to that target
#then draw the edges as black pixels on a white background
def aug_sketchify(numI):
    ksizeA = SKETCH_BLUR_KSIZE_A
    ksizeB = SKETCH_BLUR_KSIZE_B
    edge_density_multiplier = SKETCH_EDGE_DENSITY_MULTIPLIER
    canny_high_thresholds = SKETCH_CANNY_HIGH_THRESHOLDS
    canny_ratio = SKETCH_CANNY_RATIO

    #use Otsu to binarize the image to get a hint on what the edge-density should be
    numIgray = cv2.cvtColor(numI, cv2.COLOR_BGR2GRAY)
    numIblur = cv2.GaussianBlur(numIgray, (ksizeA, ksizeA), 0)
    _, numIbin = cv2.threshold(numIblur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    numIbinEdge = cv2.Canny(numIbin, 200, 100) #threshold doesn't really matter here
    target_edge_density = edge_density_multiplier * np.sum(numIbinEdge > 0) / (numIbinEdge.shape[0] * numIbinEdge.shape[1])

    #find the canny threshold that gets closest to that target
    numIblur = cv2.GaussianBlur(numIgray, (ksizeB, ksizeB), 0)
    best_density_absdiff = float('+inf')
    best_edge_mask = None
    for t in canny_high_thresholds:
        edge_mask = (cv2.Canny(numIblur, t, canny_ratio * t) > 0)
        density_absdiff = np.fabs(np.sum(edge_mask) / (edge_mask.shape[0] * edge_mask.shape[1]) - target_edge_density)
        if density_absdiff < best_density_absdiff:
            best_density_absdiff = density_absdiff
            best_edge_mask = edge_mask

    #draw edges as black on white
    numIaug = 255 * np.ones_like(numI)
    numIaug[edge_mask,:] = (0,0,0)

    return numIaug
'''

#this is an augmentation
#use (x,y) *= 1 + K * r^2 and remap
def aug_fisheye(numI):
    Ks = FISHEYE_KS
    scale_compensation_factor = FISHEYE_SCALE_COMPENSATION_FACTOR
    X, Y = np.meshgrid(np.arange(numI.shape[1]), np.arange(numI.shape[0]))
    X = X - numI.shape[1] / 2.0
    Y = Y - numI.shape[0] / 2.0
    X = X / min(numI.shape[0], numI.shape[1])
    Y = Y / min(numI.shape[0], numI.shape[1])
    R2 = np.square(X) + np.square(Y)
    distortion_multiplier = 1 + sum([K * np.float_power(R2, p * np.ones_like(R2)) for K, p in zip(Ks, [1,2,3])])
    X = distortion_multiplier * X
    Y = distortion_multiplier * Y
    X = min(numI.shape[0], numI.shape[1]) * X
    Y = min(numI.shape[0], numI.shape[1]) * Y
    X = X + numI.shape[1] / 2.0
    Y = Y + numI.shape[0] / 2.0
    numIaug = cv2.remap(numI, X.astype('float32'), Y.astype('float32'), cv2.INTER_LINEAR)
    numIaug = aug_closeup(numIaug, factor=scale_compensation_factor)
    return numIaug

#this is an augmentation
#reduce number of bits so there's fewer unique colors
def aug_posterize(numI):
    bits = POSTERIZATION_BITS
    return albumentations.augmentations.transforms.Posterize(num_bits=bits, always_apply=True)(image=numI)['image']

#this is an augmentation
#makes it grayscale
#(yes, it expects BGR)
def aug_grayscale(numI):
    numIgray = cv2.cvtColor(numI, cv2.COLOR_BGR2GRAY)
    numIgray = np.tile(numIgray[:,:,np.newaxis], (1,1,3))
    return numIgray

#this is an augmentation
#sepia-tone
#(yes, this function expects BGR. It flips the channels before feeding to albumentations.
#This is the only albumentation we use that cares about channel order.)
def aug_sepia(numI):
    numIrgb = cv2.cvtColor(numI, cv2.COLOR_BGR2RGB)
    numIsepia = albumentations.augmentations.transforms.ToSepia(always_apply=True)(image=numIrgb)['image']
    numIsepia = cv2.cvtColor(numIsepia, cv2.COLOR_RGB2BGR)
    return numIsepia

#this is an augmentation
#gaussian blur
def aug_blur(numI):
    ksize = GAUSSIAN_BLUR_KSIZE
    return cv2.GaussianBlur(numI, (ksize, ksize), 0)

#this is an augmentation
#make dimmer
def aug_dim(numI):
    factor = DIMMING_FACTOR
    return np.around(factor * numI.astype('float32')).astype('uint8')

#this is an augmentation
#make brighter
def aug_brighten(numI):
    factor = BRIGHTENING_FACTOR
    return np.around(np.minimum(factor * numI.astype('float32'), 255)).astype('uint8')

#this is an augmentation
#do a closeup (by taking a center crop)
#optional factor argument so fisheye can also make use of this function
def aug_closeup(numI, factor=None):
    if factor is None:
        factor = CLOSEUP_FACTOR #e.g. setting this to 2 would mean the crop box is half the width and half the height

    x_start = int(round((1.0 - 1.0 / factor) * numI.shape[1] / 2.0))
    x_end = int(round(x_start + numI.shape[1] / factor))
    y_start = int(round((1.0 - 1.0 / factor) * numI.shape[0] / 2.0))
    y_end = int(round(y_start + numI.shape[0] / factor))
    return copy.deepcopy(numI[y_start:y_end, x_start:x_end, :])

#this is an augmentation
#tilt +/-10-20 degrees
#note that this is a random augmentation
def aug_tilt(numI):
    min_angle = TILT_MIN_ANGLE
    max_angle = TILT_MAX_ANGLE
    angle = random.choice([1, -1]) * random.uniform(min_angle, max_angle)
    R = cv2.getRotationMatrix2D((numI.shape[1] / 2, numI.shape[0] / 2), angle, 1.0)
    return cv2.warpAffine(numI, R, (numI.shape[1], numI.shape[0]))

#this is an augmentation
#sideways (+/-90 degrees)
#note that this is random cuz there are 2 possible sides
def aug_sideways(numI):
    return np.rot90(copy.deepcopy(numI), k=random.choice([1,3]))

#this is an augmentation
#upside-down (180 degrees)
def aug_upside_down(numI):
    return np.rot90(copy.deepcopy(numI), k=2)

#this is an augmentation
#downscale, then rescale back up so it looks low-res
def aug_low_res(numI):
    factor = LOW_RES_FACTOR
    return albumentations.augmentations.transforms.Downscale(scale_min=factor, scale_max=factor, always_apply=True)(image=numI)['image']

#this is an augmentation
#adds fog
def aug_foggy(numI):
    coef = FOG_COEF
    alpha = FOG_ALPHA
    return albumentations.augmentations.transforms.RandomFog(fog_coef_lower=coef, fog_coef_upper=coef, alpha_coef=alpha, always_apply=True)(image=numI)['image']

#convenience function to generate a dictionary mapping the augmentationID to augmentation function
def generate_image_aug_dict():
    aug_dict = {}
    aug_dict['noop'] = aug_noop
    aug_dict['make_background_white'] = aug_make_background_white
    aug_dict['make_background_black'] = aug_make_background_black
    aug_dict['make_background_blue'] = aug_make_background_blue
    aug_dict['sketchify'] = aug_sketchify
    aug_dict['fisheye'] = aug_fisheye
    aug_dict['posterize'] = aug_posterize
    aug_dict['grayscale'] = aug_grayscale
    aug_dict['sepia'] = aug_sepia
    aug_dict['blur'] = aug_blur
    aug_dict['dim'] = aug_dim
    aug_dict['brighten'] = aug_brighten
    aug_dict['closeup'] = aug_closeup
    aug_dict['tilt'] = aug_tilt
    aug_dict['sideways'] = aug_sideways
    aug_dict['upside_down'] = aug_upside_down
    aug_dict['low_res'] = aug_low_res
    aug_dict['foggy'] = aug_foggy
    return aug_dict

#call this to get an idea of what the augmentations look like
if __name__ == '__main__':
    aug_dict = generate_image_aug_dict()
    image_dir = random.choice(sorted(glob.glob('../vislang-domain-exploration-data/ILSVRC2012_val/*')))
    image = random.choice(sorted(glob.glob(os.path.join(image_dir, '*.JPEG'))))
    out_dir = 'example_augs_better'
    os.makedirs(out_dir, exist_ok=True)
    numI = cv2.imread(image)
    for augID in tqdm(sorted(aug_dict.keys())):
        aug_fn = aug_dict[augID]
        numIaug = aug_fn(numI)
        cv2.imwrite(os.path.join(out_dir, os.path.splitext(os.path.basename(image))[0] + '-' + augID + '.png'), numIaug)
