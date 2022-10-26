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

        #whether to use domain classifier or something else for sampling
        self.sampling_method = 'classifier'
        self.english_only = False

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
        self.domain_classifier_inference_modality = 'image'
        self.num_laion_images_per_domain = 140000

        #clip finetuning
        self.clip_finetuning_do_disentanglement = False
        self.clip_finetuning_batch_type = 'domain_pure'
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

class NormalBatchingParams(TextTrainedDomainBalanceParams):
    ''' This is a baseline to make sure we've got the right finetuning hyperparams '''
    ''' It will also be useful for the disentanglement learning '''
    def __init__(self):
        super(NormalBatchingParams, self).__init__()

        #getting a uniform subset of LAION
        self.sampling_method = 'uniform'
        self.uniform_sample_size = 18 * 140000
        self.laion_sampling_seed = 0

        #finetuning
        self.clip_finetuning_batch_type = 'normal'

class NormalBatchingParamsLR4(NormalBatchingParams):
    def __init__(self):
        super(NormalBatchingParamsLR4, self).__init__()
        self.clip_learning_rate = 1e-4
        self.data_compatible_params_keys = ['NormalBatchingParams']

class NormalBatchingParamsLR5(NormalBatchingParams):
    def __init__(self):
        super(NormalBatchingParamsLR5, self).__init__()
        self.clip_learning_rate = 1e-5
        self.data_compatible_params_keys = ['NormalBatchingParams']

class NormalBatchingParamsLR6(NormalBatchingParams):
    def __init__(self):
        super(NormalBatchingParamsLR6, self).__init__()
        self.clip_learning_rate = 1e-6
        self.data_compatible_params_keys = ['NormalBatchingParams']

class NormalBatchingParamsLR7(NormalBatchingParams):
    def __init__(self):
        super(NormalBatchingParamsLR7, self).__init__()
        self.clip_learning_rate = 1e-7
        self.data_compatible_params_keys = ['NormalBatchingParams']

class NormalBatchingParamsLR5e8(NormalBatchingParams):
    def __init__(self):
        super(NormalBatchingParamsLR5e8, self).__init__()
        self.clip_learning_rate = 5e-8
        self.data_compatible_params_keys = ['NormalBatchingParams']

class NormalBatchingParamsLR8(NormalBatchingParams):
    def __init__(self):
        super(NormalBatchingParamsLR8, self).__init__()
        self.clip_learning_rate = 1e-8
        self.data_compatible_params_keys = ['NormalBatchingParams']

class EnglishOnlyNormalBatchingParams(NormalBatchingParams):
    def __init__(self):
        super(EnglishOnlyNormalBatchingParams, self).__init__()
        self.english_only = True

class EnglishOnlyNormalBatchingParamsLR4(EnglishOnlyNormalBatchingParams):
    def __init__(self):
        super(EnglishOnlyNormalBatchingParamsLR4, self).__init__()
        self.clip_learning_rate = 1e-4
        self.data_compatible_params_keys = ['EnglishOnlyNormalBatchingParams']
        self.english_only = True

class EnglishOnlyNormalBatchingParamsLR5(EnglishOnlyNormalBatchingParams):
    def __init__(self):
        super(EnglishOnlyNormalBatchingParamsLR5, self).__init__()
        self.clip_learning_rate = 1e-5
        self.data_compatible_params_keys = ['EnglishOnlyNormalBatchingParams']
        self.english_only = True

class EnglishOnlyNormalBatchingParamsLR6(EnglishOnlyNormalBatchingParams):
    def __init__(self):
        super(EnglishOnlyNormalBatchingParamsLR6, self).__init__()
        self.clip_learning_rate = 1e-6
        self.data_compatible_params_keys = ['EnglishOnlyNormalBatchingParams']
        self.english_only = True

class EnglishOnlyNormalBatchingParamsLR7(EnglishOnlyNormalBatchingParams):
    def __init__(self):
        super(EnglishOnlyNormalBatchingParamsLR7, self).__init__()
        self.clip_learning_rate = 1e-7
        self.data_compatible_params_keys = ['EnglishOnlyNormalBatchingParams']
        self.english_only = True

class EnglishOnlyNormalBatchingParamsLR5e8(EnglishOnlyNormalBatchingParams):
    def __init__(self):
        super(EnglishOnlyNormalBatchingParamsLR5e8, self).__init__()
        self.clip_learning_rate = 5e-8
        self.data_compatible_params_keys = ['EnglishOnlyNormalBatchingParams']
        self.english_only = True

class EnglishOnlyNormalBatchingParamsLR8(EnglishOnlyNormalBatchingParams):
    def __init__(self):
        super(EnglishOnlyNormalBatchingParamsLR8, self).__init__()
        self.clip_learning_rate = 1e-8
        self.data_compatible_params_keys = ['EnglishOnlyNormalBatchingParams']
        self.english_only = True

class DisentanglementParamsLR5e8Lambda0_1(NormalBatchingParams):
    def __init__(self):
        super(DisentanglementParamsLR5e8Lambda0_1, self).__init__()
        self.data_compatible_params_keys = ['NormalBatchingParams']
        self.clip_learning_rate = 5e-8

        #disentanglement params
        self.clip_finetuning_do_disentanglement = True
        self.disentanglement_modality = 'text'
        self.disentanglement_initialization_batch_size = 1024
        self.disentanglement_batch_size = 128
        self.disentanglement_component_optimizer_type = 'Adam'
        self.disentanglement_component_learning_rate = 1e-3
        self.disentanglement_component_scheduler_type = 'none'
        self.disentanglement_lambda = 0.1

class DisentanglementParamsLR7Lambda0_1(DisentanglementParamsLR5e8Lambda0_1):
    def __init__(self):
        super(DisentanglementParamsLR7Lambda0_1, self).__init__()
        self.data_compatible_params_keys = ['NormalBatchingParams']
        self.clip_learning_rate = 1e-7

class DisentanglementParamsLR6Lambda0_1(DisentanglementParamsLR5e8Lambda0_1):
    def __init__(self):
        super(DisentanglementParamsLR6Lambda0_1, self).__init__()
        self.data_compatible_params_keys = ['NormalBatchingParams']
        self.clip_learning_rate = 1e-6

class DisentanglementParamsLR8Lambda0_1(DisentanglementParamsLR5e8Lambda0_1):
    def __init__(self):
        super(DisentanglementParamsLR8Lambda0_1, self).__init__()
        self.data_compatible_params_keys = ['NormalBatchingParams']
        self.clip_learning_rate = 1e-8

class DisentanglementParamsLR5e8Lambda1_0(DisentanglementParamsLR5e8Lambda0_1):
    def __init__(self):
        super(DisentanglementParamsLR5e8Lambda1_0, self).__init__()
        self.data_compatible_params_keys = ['NormalBatchingParams']
        self.disentanglement_lambda = 1.0

class DisentanglementParamsLR7Lambda1_0(DisentanglementParamsLR5e8Lambda0_1):
    def __init__(self):
        super(DisentanglementParamsLR7Lambda1_0, self).__init__()
        self.data_compatible_params_keys = ['NormalBatchingParams']
        self.clip_learning_rate = 1e-7
        self.disentanglement_lambda = 1.0

class DisentanglementParamsLR6Lambda1_0(DisentanglementParamsLR5e8Lambda0_1):
    def __init__(self):
        super(DisentanglementParamsLR6Lambda1_0, self).__init__()
        self.data_compatible_params_keys = ['NormalBatchingParams']
        self.clip_learning_rate = 1e-6
        self.disentanglement_lambda = 1.0

class DisentanglementParamsLR8Lambda1_0(DisentanglementParamsLR5e8Lambda0_1):
    def __init__(self):
        super(DisentanglementParamsLR8Lambda1_0, self).__init__()
        self.data_compatible_params_keys = ['NormalBatchingParams']
        self.clip_learning_rate = 1e-8
        self.disentanglement_lambda = 1.0

class EnglishOnlyDisentanglementParamsLR5e8Lambda0_1(DisentanglementParamsLR5e8Lambda0_1):
    def __init__(self):
        super(EnglishOnlyDisentanglementParamsLR5e8Lambda0_1, self).__init__()
        self.data_compatible_params_keys = ['EnglishOnlyNormalBatchingParams']
        self.english_only = True

class EnglishOnlyDisentanglementParamsLR7Lambda0_1(EnglishOnlyDisentanglementParamsLR5e8Lambda0_1):
    def __init__(self):
        super(EnglishOnlyDisentanglementParamsLR7Lambda0_1, self).__init__()
        self.data_compatible_params_keys = ['EnglishOnlyNormalBatchingParams']
        self.clip_learning_rate = 1e-7
        self.english_only = True

class EnglishOnlyDisentanglementParamsLR5e8Lambda1_0(EnglishOnlyDisentanglementParamsLR5e8Lambda0_1):
    def __init__(self):
        super(EnglishOnlyDisentanglementParamsLR5e8Lambda1_0, self).__init__()
        self.data_compatible_params_keys = ['EnglishOnlyNormalBatchingParams']
        self.disentanglement_lambda = 1.0
        self.english_only = True

class EnglishOnlyDisentanglementParamsLR7Lambda1_0(EnglishOnlyDisentanglementParamsLR5e8Lambda0_1):
    def __init__(self):
        super(EnglishOnlyDisentanglementParamsLR7Lambda1_0, self).__init__()
        self.data_compatible_params_keys = ['EnglishOnlyNormalBatchingParams']
        self.clip_learning_rate = 1e-7
        self.disentanglement_lambda = 1.0
        self.english_only = True

class EnglishOnlyDisentanglementParamsLR6Lambda0_1(EnglishOnlyDisentanglementParamsLR5e8Lambda0_1):
    def __init__(self):
        super(EnglishOnlyDisentanglementParamsLR6Lambda0_1, self).__init__()
        self.data_compatible_params_keys = ['EnglishOnlyNormalBatchingParams']
        self.clip_learning_rate = 1e-6
        self.english_only = True

class EnglishOnlyDisentanglementParamsLR8Lambda0_1(EnglishOnlyDisentanglementParamsLR5e8Lambda0_1):
    def __init__(self):
        super(EnglishOnlyDisentanglementParamsLR8Lambda0_1, self).__init__()
        self.data_compatible_params_keys = ['EnglishOnlyNormalBatchingParams']
        self.clip_learning_rate = 1e-8
        self.english_only = True

class TextTrainedTextInferenceDomainBalanceParams(TextTrainedDomainBalanceParams):
    def __init__(self):
        super(TextTrainedTextInferenceDomainBalanceParams, self).__init__()
        self.domain_classifier_inference_modality = 'text'

class TextMatchingDomainBalanceParams(TextTrainedDomainBalanceParams):
    def __init__(self):
        super(TextMatchingDomainBalanceParams, self).__init__()

        #use simple text-matching (on LAION captions) instead of domain classifier for sampling
        self.sampling_method = 'text_matching'

        #params for sampling
        self.laion_sampling_seed = 0
        self.laion_sampling_safety_factor = 5

class ImageTrainedDomainBalanceParams(TextTrainedDomainBalanceParams):
    def __init__(self):
        super(ImageTrainedDomainBalanceParams, self).__init__()

        #use image embeddings instead of text embeddings to train the domain classifier
        #just use one training image per (domain, class) pair
        #(this is what most papers would report as "18-shot", although from a domain-classification perspective it's actually 1000-shot)
        self.domain_classifier_train_embedding_type = 'image'
        self.domain_classifier_image_shots_per_domainclass = 1
        self.domain_classifier_image_sampling_seed = 0

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
