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

As a first step, let's try and explore the embedding space of CLIP with some handcrafted inputs, just to see how it represents domains. And let's try out one initial idea at improving the DG performance of CLIP.  


### 7/22/2022 - 9/30/2022  
  
(Note: Look in the `job_scripts` folder to see bash scripts that run the python scripts described below. That should give a good record of the experiments that happened and how to replicate them.)  
  
**Probing - data fundamentals**  
* For probing (and the initial DG attempt), we create 18 different domains by applying 17 different augmentations to images and describing them in text.  
* `image_aug_utils.py` handles the image side, and `text_aug_utils.py` handles the text side. `general_aug_utils.py` calls on both of these to handle both sides.  
* For probing (and testing of initial DG attempt), we apply our augmentations to the ImageNet1K validation set. We also use the ImageNet1K training set for linear probing. For accessing this dataset, please see `non_image_data_utils.py`, specifically the function `load_non_image_data()`. For `base_dir` you should pass in the path to `ILSVRC2012_val` or `ILSVRC_train`. Each of these folders should have a file inside called `words.txt`, which can be gotten [here](https://github.com/seshuad/IMagenet/blob/master/tiny-imagenet-200/words.txt).  
  
**Probing - CLIP embeddings and ChunkWriter**  
* `compute_CLIP_embeddings.py` will compute the CLIP embeddings for all the augmented images and all the textual descriptions of the augmentations (paired with the classes). There's a flag that can make it compute only image embeddings, and a way to make it start at a particular place (so you could run lots of processes in parallel and have each one start from a different place, skipping over stuff that's already been done).  
* `chunk_writer.py` will take care of reading *and* writing an embedding dict in shards. You make one for images, and/or one for text. The one catch is that you do have to provide all the keys in advance, so it can figure out which shard each key goes into. I guess one advantage of this is that you can access entries for just one key or a few keys, without having to load all the shards.  
  
**Probing - analysis**
* `do_robustness_and_recovery_analysis.py` will compute the decrease in cosine similarity and zero-shot accuracy when augmenting the image, and the recovery when describing those augmentations in text. Functions like `compute_zeroshot_preds()` and `is_it_correct()` are also used by the initial DG idea for prediction/evaluation.  
* `do_official_CLIP_zeroshot_analysis.py` (with the help of `CLIP_paper_official_zeroshot_text_utils.py`) does the zeroshot decrease analysis using the "official" CLIP text embeddings, which are an average of embeddings with a diverse set of text templates.  
* For linear probing analysis, take a look at `fit_linear_probes.py` and `evaluate_linear_probes.py`.  
* For entanglement analysis, take a look at `do_subspace_analysis.py`.  
* Most/all of these analysis scripts save their results as dicts in `.pkl` files. In order to turn these into plots and tables, take a look at `tablify_robustness_and_recovery_stats_dict.py`, `tablify_official_CLIP_zeroshot_stats_dict.py`, `tablify_linearprobe_stats_dict.py`, `tablify_subspace_stats_dict.py`, `make_robustness_plots.py`, `make_linearprobe_plot.py`, and `make_subspace_plots.py`. Most/all of these use `plot_and_table_utils.py`.  
  
**Domain Generalization - LAION setup**  
* Idea was to use text embeddings to train a domain classifier, then use it to sample domain-pure subsets from LAION (using the provided LAION image embeddings). Fine-tune OpenAI CLIP checkpoint with domain-pure LAION batches.  
* First step is to download image embeddings and metadata of a very large subset of LAION. Do this by picking shards at random to get a total of 250M datapoints. Same proportion of LAION-en, LAION-multi, LAION-nolang. Use `download_laion5b_subset_embeddings.py`.  
* Simplify the metadata into a (sharded) info dict, using `make_laion_image_level_info_dict.py`.  
  
**Domain Generalization - config**  
* See `experiment_params/balance_params.py` for config classes. An instance of one of these classes will carry all the config stuff. `grab_params()` can construct one of these instances given its name (the "params_key"). See `experiment_params/param_utils.py` for utility functions related to this.  
  
**Domain Generalization - domain classifier**  
* `train_domain_classifier.py` will train a domain classifier given texts describing the domains (and classes of objects). It uses `embedding_domain_and_class_dataset.py` for a custom training Dataset.  
* To benchmark on augmented ImageNet1K validation images, take a look at   
* sdfsdf  
  
