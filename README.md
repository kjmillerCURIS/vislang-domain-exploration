# vislang-domain-exploration

## Description

The purpose of this project is to use language as a signal for visual domain adaptation/generalization. This is a particularly interesting research direction given the rise of large vision-language-pretrained models like CLIP.

## Installation

Unfortunately I am unable to recreate the conda environment from a requirements.txt from either pip or conda, so here are the commands to run to set it up. 


`conda create --name vislang-domain-exploration python=3.8`

`conda activate vislang-domain-exploration`

`pip install git+https://github.com/openai/CLIP.git`

`pip install opencv-python scipy matplotlib pandas albumentations ipython`

`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`

`pip install pyarrow fastparquet`

`pip install img2dataset` (I don't remember if I ran this command. But img2dataset was on my system, so either I installed it or someone/something else did.)

`conda install -c conda-forge pytorch-lightning lightning-bolts`

Then, go to this line in the img2dataset code on your system (https://github.com/rom1504/img2dataset/blob/c0f14c9020003483f9b30b960317187f9a6c6b97/img2dataset/reader.py#L72) and change it to use '\t' as the delimiter (instead of the default ','). 

[TODO: UPDATE requirement FILES!!!]
Note that these instructions don't specify version. For version info, look at `requirements_pip.txt` and `requirements_conda.txt`.

## Exploratory Probing and Domain Generalization

As a first step, let's try and explore the embedding space of CLIP with some handcrafted inputs, just to see how it represents domains.

### 7/22/2022 - 9/30/2022  
  
**Probing - data fundamentals**  
* For probing (and the initial DG attempt), we create 18 different domains by applying 17 different augmentations to images and describing them in text.  
* `image_aug_utils.py` handles the image side, and `text_aug_utils.py` handles the text side. `general_aug_utils.py` calls on both of these to handle both sides.  
* For probing (and testing of initial DG attempt), we apply our augmentations to the ImageNet1K validation set. We also use the ImageNet1K training set for linear probing. For accessing this dataset, please see `non_image_data_utils.py`, specifically the function `load_non_image_data()`. For `base_dir` you should pass in the path to `ILSVRC2012_val` or `ILSVRC_train`. Each of these folders should have a file inside called `words.txt`, which can be gotten [here](https://github.com/seshuad/IMagenet/blob/master/tiny-imagenet-200/words.txt).  
* Meow.  

