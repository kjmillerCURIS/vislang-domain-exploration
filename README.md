# vislang-domain-exploration

## Description

The purpose of this project is to use language as a signal for visual domain adaptation/generalization. This is a particularly interesting research direction given the rise of large vision-language-pretrained models like CLIP.

## Installation

Unfortunately I am unable to recreate the conda environment from a requirements.txt from either pip or conda, so here are the commands to run to set it up. 


`conda create --name vislang-domain-exploration python=3.8`

`conda activate vislang-domain-exploration`

`pip install git+https://github.com/openai/CLIP.git`

`pip install opencv-python`


Note that these instructions don't specify version. For version info, look at `requirements_pip.txt` and `requirements_conda.txt`.

## Exploratory Probing

As a first step, let's try and explore the embedding space of CLIP with some handcrafted inputs, just to see how it represents domains.

### 7/22/2022 - Start of domain-probing (some development done in Google Colab before)

Idea is to take 
