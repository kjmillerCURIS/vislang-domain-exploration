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
#this generator will go on forever, but once every "epoch" it will yield None to let the user know that an "epoch" has passed
#the "epoch" is defined as calling each dataloader N times, where N is the average number of batches in each of them
#this might not line up with individual epochs of the dataloaders if they have different numbers of batches
#it will also yield the domain index of the batch. Specifically, it will yield (batch, domain_index) or (None, None)
def dataloader_multiplexer(dataloaders):
    num_cycles_per_epoch = int(round(sum([len(dataloader) for dataloader in dataloaders]) / len(dataloaders)))
    forevers = [forever_dataloader(dataloader) for dataloader in dataloaders]
    while True:
        for t_cycle in range(num_cycles_per_epoch):
            indices = random.sample(range(len(forevers)), len(forevers)) #this is equivalent to shuffling
            for index in indices:
                yield (next(forevers[index]), index)

        yield (None, None) #sentinel to let the user know an "epoch" has passed
