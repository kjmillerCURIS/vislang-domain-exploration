import os
import sys
import random

#make a generator that yields batches from dataloader forever and ever and ever...
def forever_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

#this is a generator
#it will yield a batch from each dataloader in dataloaders, cycling through them in a different (random) order each time
#it will also yield the domain index along with the batch
#this generator has no concept of an "epoch". It will NOT stop. It will NOT give any sentinel indicating that an epoch has passed.
#it is the caller's responsibility to decide how many calls counts as an epoch
def dataloader_multiplexer(dataloaders):
    forevers = [forever_dataloader(dataloader) for dataloader in dataloaders]
    while True:
        indices = random.sample(range(len(forevers)), len(forevers)) #this is equivalent to shuffling
        for index in indices:
            yield next(forevers[index]), index
