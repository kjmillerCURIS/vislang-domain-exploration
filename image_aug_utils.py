import os
import sys
import copy
import cv2
import numpy as np
import random
import albumentations

SEG_BBOX_FACTOR = 0.5 #how much to shrink the image frame to get the bbox
SEG_CONTRACTION_ALPHA = 0.25 #how much to shrink the bbox to get the definitely-foreground part
SEG_GC_NUM_ITERS = 5

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
    #make a central bbox
    bbox_factor = SEG_BBOX_FACTOR
    w = int(round((1 - bbox_factor) * numI.shape[1]))
    h = int(round((1 - bbox_factor) * numI.shape[0]))
    x = (numI.shape[1] - w) // 2
    y = (numI.shape[0] - h) // 2

    #anything outside the bbox is definite background
    #anything inside a contracted version of the definite foreground
    #anything else is probable foreground
    contraction_alpha = SEG_CONTRACTION_ALPHA
    mask = np.zeros(numI.shape[:2],np.uint8)
    mask[:,:] = cv2.GC_BGD
    mask[x:x+w, y:y+h] = cv2.GC_PR_FGD
    x_start = int(x + contraction_alpha * w)
    x_end = int(x + (1 - contraction_alpha) * w)
    y_start = int(y + contraction_alpha * h)
    y_end = int(y + (1 - contraction_alpha) * h)
    mask[y_start:y_end, x_start:x_end] = cv2.GC_FGD

    #do grabcut
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(numI,mask,None,bgdModel,fgdModel,SEG_GC_NUM_ITERS,cv2.GC_INIT_WITH_MASK)
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
#use Otsu to get a hint on what the edge-density should be like
#then adjust Canny threshold to get closest to that target
#then draw the edges as black pixels on a white background
def aug_sketchify(numI):
    ksize = SKETCH_BLUR_KSIZE
    edge_density_multiplier = SKETCH_EDGE_DENSITY_MULTIPLIER
    canny_high_thresholds = SKETCH_CANNY_HIGH_THRESHOLDS
    canny_ratio = SKETCH_CANNY_RATIO

    #use Otsu to binarize the image to get a hint on what the edge-density should be
    numIgray = cv2.cvtColor(numI, cv2.COLOR_BGR2GRAY)
    numIblur = cv2.GaussianBlur(numIgray, (ksize, ksize), 0)
    _, numIbin = cv2.threshold(numIblur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    numIbinEdge = cv2.Canny(numIbin, 200, 100) #threshold doesn't really matter here
    target_edge_density = edge_density_multiplier * np.sum(numIbinEdge > 0) / (numIbinEdge.shape[0] * numIbinEdge.shape[1])

    #find the canny threshold that gets closest to that target
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

#this is an augmentation
#use (x,y) *= 1 + K * r^2 and remap
def aug_fisheye(numI):
    K = FISHEYE_K
    X, Y = np.meshgrid(numI.shape[1], numI.shape[0])
    X -= numI.shape[1] / 2
    Y -= numI.shape[0] / 2
    X /= min(numI.shape[0], numI.shape[1])
    Y /= min(numI.shape[0], numI.shape[1])
    distortion_multiplier = 1.0 + K * (np.square(X) + np.square(Y))
    X *= distortion_multiplier
    Y *= distortion_multiplier
    numIaug = cv2.remap(numI, X, Y, cv2.INTER_LINEAR)
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
    return cv2.GaussianBlur(numI, (ksize, ksize))

#this is an augmentation
#make dimmer
def aug_dim(numI):
    factor = DIMMING_FACTOR
    return np.around(factor * numI).astype('uint8')

#this is an augmentation
#make brighter
def aug_brighten(numI):
    factor = BRIGHTENING_FACTOR
    return np.around(np.maximum(factor * numI, 255)).astype('uint8')

#this is an augmentation
#do a closeup (by taking a center crop)
def aug_closeup(numI):
    factor = CLOSEUP_FACTOR #e.g. setting this to 2 would mean the crop box is half the width and half the height
    x_start = int(round((1.0 - 1.0 / factor) * numI.shape[1] / 2.0))
    x_end = int(round(x_start + numI.shape[1] / factor))
    y_start = int(round((1.0 - 1.0 / factor) * numI.shape[0] / 2.0))
    y_end = int(round(x_start + numI.shape[0] / factor))
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

#convenience function to generate a dictionary mapping the augmentationID to augmentation function
def generate_aug_dict():
    assert(False)

#call this to get an idea of what the augmentations look like
if __name__ == '__main__':
    my_aug_dict = generate_aug_dict()
    image_dir = random.choice(sorted(glob.glob('../vislang-domain-exploration-data/ILSVRC2012_val/*')))
