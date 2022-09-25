import os
import sys
import torch
from chunk_writer import ChunkWriter
from non_image_data_utils import load_non_image_data

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

    def __get_domain_filter_from_data(self, embedding_dict, embedding_type):
        if embedding_type == 'image':
            return sorted(set([k[1] for k in sorted(embedding_dict.keys())]))
        elif embedding_type == 'text':
            return sorted(set([k[2] for k in sorted(embedding_dict.keys())]))
        else:
            assert(False)

    def __get_class_filter_from_data(self, embedding_dict, embedding_type, image2class_dict=None):
        if embedding_type == 'image':
            return sorted(set([image2class_dict[k[0]] for k in sorted(embedding_dict.keys())]))
        elif embedding_type == 'text':
            return sorted(set([k[0] for k in sorted(embedding_dict.keys())]))
        else:
            assert(False)

    #embedding_dict_filename_prefix, embedding_type get passed to ChunkWriter
    #domain_filter should be a list of domains to allow. If None, then allow all domains.
    #class_filter does the same thing, but for classes
    #whatever list of domains/classes we get, we will sort it by alphabetical order and use that for indexing
    #(yes, we will hold on to a copy of this sorted list)
    #base_dir should be passed in if embedding_type is 'image'. It will be used to get things like class2filenames_dict.
    #each sample will have keys 'embedding', 'domain', 'class' (unless/until otherwise specified)
    #(yes, this class WILL store all the embeddings in (CPU) RAM at once)
    def __init__(self, embedding_dict_filename_prefix, embedding_type, domain_filter=None, class_filter=None, base_dir=None):
        assert(embedding_type in ['image', 'text'])
        my_chunk_writer = ChunkWriter(None, None, embedding_type, [], embedding_dict_filename_prefix, readonly=True)
        embedding_dict = my_chunk_writer.load_entire_dict(target_dtype='float32')

        image2class_dict = None
        if embedding_type == 'image':
            image2class_dict = self.__get_image2class_dict(base_dir)

        if domain_filter is None:
            domain_filter = self.__get_domain_filter_from_data(embedding_dict, embedding_type)

        if class_filter is None:
            class_filter = self.__get_class_filter_from_data(embedding_dict, embedding_type, image2class_dict=image2class_dict)

        self.samples = []
        for k in sorted(embedding_dict.keys()):
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

            embedding_size = len(embedding_dict[k])
            self.samples.append({'embedding' : embedding_dict[k], 'domain' : torch.tensor(domain_filter.index(domain), dtype=torch.long), 'class' : torch.tensor(class_filter.index(my_class), dtype=torch.long)})

        self.domain_filter = domain_filter
        self.class_filter = class_filter
        self.embedding_size = embedding_size

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
