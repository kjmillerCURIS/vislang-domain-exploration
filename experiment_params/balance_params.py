import os
import sys

class TextTrainedDomainBalanceParams:
    ''' use synthetic augmentations for domains '''
    ''' use text embeddings to train the domain classifier '''
    ''' use vanilla CLIP finetuning with *only* same-domain batches '''
    ''' use all the domains and classes to create texts for training the domain classifier, and test on all of them '''
    ''' (will do a cross-validation or leave-one-out over the domains later, and maybe also an "unseen classes" experiment) '''
    ''' for now, testing can be zero-shot classification with both standard text and augmented text, to see if either one of them improves '''

    def __init__(self):
        self.domain_type = 'synthetic'
        self.domain_split_type = 'all_all'
        self.class_split_type = 'all_all'

        #domain classifier training data
        self.domain_classifier_train_embedding_type = 'text'

        #domain classifier architecture
        self.domain_classifier_hidden_layer_props = [0.25]
        self.domain_classifier_hidden_layer_activations = ['GELU']
        self.domain_classifier_hidden_layer_dropouts = [False]
        self.domain_classifier_hidden_layer_batchnorms = [False]
        self.domain_classifier_output_layer_dropout = False
        self.domain_classifier_include_skip_layer = True

        #domain classifier training procedure
        self.domain_classifier_batch_size = 32
        self.domain_classifier_num_epochs = 10
        self.domain_classifier_optimizer = 'Adam'
        self.domain_classifier_scheduler = 'OneCycleLR'
        self.domain_classifier_max_lr = 2e-3
        self.domain_classifier_checkpoint_freq = 1 #in epochs

        #sampling from laion
        self.num_laion_images_per_domain = 140000

class TextTrainedDomainBalanceLinearClassifierParams(TextTrainedDomainBalanceParams):
    def __init__(self):
        super(TextTrainedDomainBalanceLinearClassifierParams, self).__init__()
        self.domain_classifier_hidden_layer_props = []
        self.domain_classifier_hidden_layer_activations = []
        self.domain_classifier_hidden_layer_dropouts = []
        self.domain_classifier_hidden_layer_batchnorms = []
        self.domain_classifier_include_skip_layer = False

class TextTrainedDomainBalance100EpochsParams(TextTrainedDomainBalanceParams):
    def __init__(self):
        super(TextTrainedDomainBalance100EpochsParams, self).__init__()
        self.domain_classifier_num_epochs = 100

class TextTrainedDomainBalanceLinearClassifier100EpochsParams(TextTrainedDomainBalanceLinearClassifierParams):
    def __init__(self):
        super(TextTrainedDomainBalanceLinearClassifier100EpochsParams, self).__init__()
        self.domain_classifier_num_epochs = 100

def grab_params(params_key):
    return eval(params_key + '()')
