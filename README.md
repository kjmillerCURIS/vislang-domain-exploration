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
  
(Note: progress update from 9/30/2022 lab meeting is [here](https://docs.google.com/presentation/d/1SZ7vQfwyh6qUUWqppIAmY5XV4HLbUdbH/edit?usp=sharing&ouid=114870409551880709293&rtpof=true&sd=true))  
  
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
* `do_official_CLIP_zeroshot_analysis.py` (with the help of `CLIP_paper_official_zeroshot_text_utils.py`) does the zeroshot decrease analysis using the "official" CLIP text embeddings, which are an average of embeddings with a diverse set of text templates. Note that there's actually 2 scripts you have to run: first, you have to run `compute_CLIP_paper_zeroshot_classifier.py`, which saves the averaged text embeddings (the "classifier") as an npy file, and, and then you have to run `do_official_CLIP_zeroshot_analysis.py`.  
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
* **Note**: the LAION-provided image embeddings are with ViT-L/14, so we have to embed our training texts with ViT-L/14 in order to be able to use them. So you have to use `compute_CLIP_embeddings.py` with that CLIP model specified to do that. The next bullet-point will talk about benchmarking the domain classifier on augmented ImageNet1K validation images, so you'll want to embed those with ViT-L/14 too. This is in contrast to the previous analysis and the subsequent CLIP finetuning, which uses ViT-B/32.  
* To benchmark on augmented ImageNet1K validation images, take a look at `compute_domain_classifier_accuracy_on_images.py`. Note that this also uses `embedding_domain_and_class_dataset.py`, but to load images instead of text. It'll do 2 kinds of evaluations. First, it'll assess the accuracy of predicting the domain. Second, it'll pull the top-K scorers of each domain and look at which actual domains end up in that sample. You'll get confusion matrices for both types of evaluations.  
  
**Domain Generalization - sampling from LAION**  
* Take a look at `compute_domain_log_probs_on_laion.py`, which inferences the domain classifier on the provided LAION image embeddings and stores the domain log-probs in a (sharded) dict. It uses `laion_image_embedding_dataset.py` for the custom test Dataset. For some reason it takes an entire day to run, even when I throw 8 GPUs at it (note: I also throw 8 CPU cores at it, but I still keep num_workers=0 in the Dataloader because I don't trust my shard-caching Dataset to be thread-safe). I know it's 250M embeddings, but it's such a lightweight classifier that I suspect that I'm doing something wrong and could be making this part much faster. Maybe I should be using DistributedDataParallel instead of just DataParallel? Or just ditch parallelism altogether and just split it across multiple jobs? Or just run the model on CPU (you never know...)?  
* Take a look at `sample_from_laion.py`, which picks the top-K LAION image-text pairs by log-prob of each domain. I've set K to 150,000 per domain. This will make the "laion_sample" folder inside of the experiment_dir, and leave various .txt and .pkl files telling the image-bases and URLs and captions.  
* (A note about "image_base": I had originally assumed that I'd get to name each image file as I downloaded it, and I usually like to use image basenames as keys into all the dicts. But I found out that img2dataset just names each file as a number. So as a "compromise" I made the file <experiment_dir>/laion_sample/<domain_dir>/image_bases.pkl, which lists the image-bases in the order that they're (attemptedly) downloaded. So image_bases[i] should correspond to the image at <experiment_dir>/laion_sample/<domain_dir>/images/"%09d"%(i)[:5]/"%09d.jpg"%(i). If an image fails to download, that will NOT mess up the indexing, there will just be a gap in the filenames.)  
* Take a look at `download_images_from_laion.py`, which uses img2dataset to do the actual downloading. It's surprisingly fast. It only took half an hour for SCC to download 2.3M images (across 6 jobs with 4 CPU cores per job). I guess it's worth noting that I had img2dataset save the images as short_side=224, default jpg quality (95%), keep_ratio.  
* Use `make_collage.py` to make a "collage" of a small random sample of the downloaded images of each domain. Caution: there's currently no NSFW filtering used...  
  
**Domain Generalization - finetuning CLIP**  
* The main script for finetuning is `finetune_clip_on_domain_pure_batches.py`, which uses `laion_image_and_text_dataset_one_domain.py` as its custom Dataset. It uses `dataloader_multiplexer.py` to (randomly) rotate between the domains when sampling batches. It uses `clip_training_utils.py` to do the loss and backprop stuff, including how to handle a large batches by breaking them down into smaller minibatches in a way that still gets the gradient correct. `clip_training_utils.py` also provides the `grab_clip_backbones()` function, which can grab the image and text backbones from the OpenAI checkpoint, which is useful for both training and inference.  
* (Note: OpenAI checkpoint is natively in float16 (I'm 90% sure). For now, I finetune and inference it in float16. The Nature paper seems to get away with finetuning in float16, so hopefully that's not a crazy idea...)  
* Sometimes you've already downloaded from LAION into one experiment_dir, and then you want to change some detail of finetuning, so you need another experiment_dir for that experiment, but you just want to symlink the already-downloaded LAION data into the new experiment_dir. Here's a convenience script to do that: `copy_experiment_dir_with_data_symlink.py`  
  
**Domain Generalization - prediction and evaluation**  
* Evaluate on augmented ImageNet1K validation images, and try zero-shot predictions with both "standard" text template and own-domain text template. Haven't implemented evaluation with "official" CLIP texts yet.  
* Can evaluate on any saved checkpoint, including "fractional" checkpoints, which is useful because currently the results SUCK by the end of even the first epoch, but they do have a slight improvement, like ~0.8%, *very* early on...  
* First step is to embed the images and texts with the finetuned backbones. Use `inference_clip_checkpoints.py` for this.  
* Then, run zero-shot classification and compute accuracy. Use `evaluate_zeroshot_with_checkpoints.py` for this.  
* Finally, plot the results with `make_clip_finetuning_plots.py`  
  
