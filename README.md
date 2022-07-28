# vislang-domain-exploration

## Description

The purpose of this project is to use language as a signal for visual domain adaptation/generalization. This is a particularly interesting research direction given the rise of large vision-language-pretrained models like CLIP.

## Installation

Unfortunately I am unable to recreate the conda environment from a requirements.txt from either pip or conda, so here are the commands to run to set it up. 


`conda create --name vislang-domain-exploration python=3.8`

`conda activate vislang-domain-exploration`

`pip install git+https://github.com/openai/CLIP.git`

`pip install opencv-python scipy matplotlib pandas albumentations ipython`


Note that these instructions don't specify version. For version info, look at `requirements_pip.txt` and `requirements_conda.txt`.

## Exploratory Probing

As a first step, let's try and explore the embedding space of CLIP with some handcrafted inputs, just to see how it represents domains.

### 7/22/2022 - Start of domain-probing (some development done in Google Colab before)

NOTE TO SELF: DO NOT UNDER ANY CIRCUMSTANCES USE THE IMAGENET VALIDATION SET AS TRAINING DATA FOR A "SOTA" DISENTANGLEMENT-LEARNING ALGO!!!

(Also note: you technically could do disentanglement-learning with only-images or only-text...oh, and for the "don't collapse" loss, you could just try and reconstruct the original embedding, probably with some VAE-noising-type strategy to make sure we're not "hiding" any reconstruction info)

We will do the following augmentations:

* white background

* black background

* blue background

* sketch/drawing (fixed-sized blur (default), then Otsu to get a hint on how many edge-pixels there should be, then multiply that number by a fixed factor (probably 1.2), then try a bunch of Canny's with fixed ratio (default) between high and low threshold, and pick the one that has the closest edge-density. Then, just make the edges black and everything else white).

* fisheye (remap with `(x, y) *= (1 + K * r^2)` where (x, y) are centered and divided by shorter dimension before computing r. Start with K=0.2 and tune it by hand)

* posterize

* grayscale

* sepia (albumentations.augmentations.transforms.ToSepia) (these may have some overlap in possible prompts)

* Gaussian blur (ksize=15)

* dimming (multiply by 0.5)

* brighting (mulitply by 1.5)

* closeup (2x)

* tilted (+/- 10-20 degrees)

* sideways

* upside-down

* low-res (down by 4x, then back up)
