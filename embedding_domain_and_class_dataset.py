import os
import sys
import random
import torch
from tqdm import tqdm
from chunk_writer import ChunkWriter
from non_image_data_utils import load_non_image_data
from compute_CLIP_embeddings import write_to_log_file

#ChunkWriter(None, None, 'image', [], embedding_dict_filename_prefix, readonly=True)
#k = (image_base, augID)
#k = (classID, className, augID, text_aug_template)

class EmbeddingDomainAndClassDataset(torch.utils.data.Dataset):

    def __get_image2class_dict(self, base_dir):
        _, __, class2filenames_dict = load_non_image_data(base_dir)
        image2class_dict = {}
        for classID in sorted(class2filenames_dict.keys()):
            for image_path in class2filenames_dict[classID]:
                image2class_dict[os.path.basename(image_path)] = classID

        return image2class_dict

    def __get_domain_filter_from_data(self, all_keys, embedding_type):
        if embedding_type == 'image':
            return sorted(set([k[1] for k in all_keys]))
        elif embedding_type == 'text':
            return sorted(set([k[2] for k in all_keys]))
        else:
            assert(False)

    def __get_class_filter_from_data(self, all_keys, embedding_type, image2class_dict=None):
        if embedding_type == 'image':
            return sorted(set([image2class_dict[k[0]] for k in all_keys]))
        elif embedding_type == 'text':
            return sorted(set([k[0] for k in all_keys]))
        else:
            assert(False)

    #returns a list of (image_base, domain) pairs that are allowed to be used
    def __make_image_domain_filter(self, all_keys, image2class_dict, image_shots_per_domainclass=1, image_sampling_seed=0):
        assert(image_shots_per_domainclass is not None)
        random.seed(image_sampling_seed)
        domainclass2list = {}
        for k in all_keys:
            image_base, domain = k
            my_class = image2class_dict[image_base]
            if (domain, my_class) not in domainclass2list:
                domainclass2list[(domain, my_class)] = []

            domainclass2list[(domain, my_class)].append((image_base, domain))

        image_domain_filter = []
        for (domain, my_class) in sorted(domainclass2list.keys()):
            image_domain_filter.extend(random.sample(domainclass2list[(domain, my_class)], image_shots_per_domainclass))

        return image_domain_filter

    #embedding_dict_filename_prefix, embedding_type get passed to ChunkWriter
    #domain_filter should be a list of domains to allow. If None, then allow all domains.
    #class_filter does the same thing, but for classes
    #whatever list of domains/classes we get, we will sort it by alphabetical order and use that for indexing
    #(yes, we will hold on to a copy of this sorted list)
    #base_dir should be passed in if embedding_type is 'image'. It will be used to get things like class2filenames_dict.
    #each sample will have keys 'embedding', 'domain', 'class' (unless/until otherwise specified)
    #(yes, this class WILL store all the embeddings in (CPU) RAM at once)
    def __init__(self, embedding_dict_filename_prefix, embedding_type, domain_filter=None, class_filter=None, base_dir=None, image_shots_per_domainclass=1, image_sampling_seed=0):
        assert(embedding_type in ['image', 'text'])
        my_chunk_writer = ChunkWriter(None, None, embedding_type, [], embedding_dict_filename_prefix, readonly=True)
        image2class_dict = None
        if embedding_type == 'image':
            image2class_dict = self.__get_image2class_dict(base_dir)

        if domain_filter is None:
            domain_filter = self.__get_domain_filter_from_data(my_chunk_writer.get_keys(), embedding_type)

        if class_filter is None:
            class_filter = self.__get_class_filter_from_data(my_chunk_writer.get_keys(), embedding_type, image2class_dict=image2class_dict)

        if embedding_type == 'image' and image_shots_per_domainclass is not None:
            image_domain_filter = self.__make_image_domain_filter(my_chunk_writer.get_keys(), image2class_dict, image_shots_per_domainclass=image_shots_per_domainclass, image_sampling_seed=image_sampling_seed)

        write_to_log_file('done getting all the filters')

        self.samples = []

        all_keys = my_chunk_writer.get_keys()
        if embedding_type == 'image' and image_shots_per_domainclass is not None:
            all_keys = sorted(image_domain_filter)

        for k in tqdm(all_keys):
            if embedding_type == 'image':
                image_base, domain = k
                my_class = image2class_dict[image_base]
            elif embedding_type == 'text':
                my_class, _, domain, __ = k
            else:
                assert(False)

            if domain not in domain_filter:
                continue

            if my_class not in class_filter:
                continue

            if not my_chunk_writer.contains(k):
                print('!!! missing key "%s" !!!'%(str(k)))
                continue

            embedding = my_chunk_writer.get(k, target_dtype='float32')
            embedding_size = len(embedding)
            self.samples.append({'embedding' : embedding, 'domain' : torch.tensor(domain_filter.index(domain), dtype=torch.long), 'class' : torch.tensor(class_filter.index(my_class), dtype=torch.long)})

        self.domain_filter = domain_filter
        self.class_filter = class_filter
        self.embedding_size = embedding_size

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
