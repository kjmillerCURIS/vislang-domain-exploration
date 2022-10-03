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

        #clip finetuning
        self.clip_model_type = 'ViT-B/32'
        self.clip_optimizer_type = 'AdamW'
        self.clip_weight_decay = 0.2 #I'm questioning whether to do weight decay when finetuning
        self.clip_learning_rate = 5e-4
        self.clip_beta1 = 0.9
        self.clip_beta2 = 0.98
        self.clip_epsilon = 1e-6
        self.clip_scheduler_type = 'LinearWarmupCosineAnnealingLR'
        self.clip_max_epochs = 10
        self.clip_warmup_epochs = 1 #for now we'll always make this 10% of the max epochs
        self.clip_batch_size = 1024
        self.clip_oversize_batch_mode = True
        self.clip_image_minibatch_size = 64
        self.clip_text_minibatch_size = 64
        self.clip_fractional_checkpoints = [1/1024, 1/512, 1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 3/4, 5/4, 3/2, 7/4]

class TextTrainedDomainBalanceSmallBatchParams(TextTrainedDomainBalanceParams):
    def __init__(self):
        super(TextTrainedDomainBalanceSmallBatchParams, self).__init__()
        self.data_compatible_params_keys = ['TextTrainedDomainBalanceParams']
        self.clip_batch_size = 64
        self.clip_oversize_batch_mode = False

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
