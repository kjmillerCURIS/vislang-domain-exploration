#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 1
#$ -l h_rt=11:59:59
#$ -N copy_imagenet_train
#$ -j y
#$ -m ea

cp -r /net/ivcfs5/mnt/data/imagenet/data/ILSVRC2012_train data

