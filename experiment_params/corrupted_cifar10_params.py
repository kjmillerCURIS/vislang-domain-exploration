import os
import sys

class CorruptedCIFAR10BaselineParamsLR5:

    def __init__(self):

        #domain classifier architecture
        self.num_layers_to_use_for_adapter = 3 #number of layers from the CLIP backbone, NOT number of layers within the adapter itself!
        self.adapter_hidden_layer_props = [0.25]
        self.adapter_hidden_layer_activations = ['GELU']
        self.adapter_hidden_layer_dropouts = [False]
        self.adapter_hidden_layer_batchnorms = [False]
        self.adapter_output_layer_dropout = False
        self.adapter_include_skip_layer = True
        self.adapter_residual_type = 'learnable'

        #clip finetuning
        self.domainless_text_prop = 0.0
        self.do_disentanglement = False
        self.do_closeness_loss = False
        self.clip_model_type = 'ViT-B/32'
        self.clip_optimizer_type = 'AdamW'
        self.clip_weight_decay = 0.2 #I'm questioning whether to do weight decay when finetuning
        self.clip_learning_rate = 1e-5
        self.clip_beta1 = 0.9
        self.clip_beta2 = 0.98
        self.clip_epsilon = 1e-6
        self.clip_scheduler_type = 'LinearWarmupCosineAnnealingLR'
        self.clip_max_epochs = 100
        self.clip_warmup_epochs = 10 #for now we'll always make this 10% of the max epochs
        self.clip_batch_size = 1024
        self.clip_oversize_batch_mode = False
        self.clip_fractional_checkpoints = [1/4, 1/2, 3/4, 5/4, 3/2, 7/4]

#try smaller batches to address weird accuracy trends

class CorruptedCIFAR10BaselineParamsLR5BatchSize64(CorruptedCIFAR10BaselineParamsLR5):

    def __init__(self):
        super(CorruptedCIFAR10BaselineParamsLR5BatchSize64, self).__init__()
        self.clip_batch_size = 64

#introduce domainless text templates to make it more challenging

class CorruptedCIFAR10BaselineParamsLR5BatchSize64DomainlessTextProp50(CorruptedCIFAR10BaselineParamsLR5BatchSize64):

    def __init__(self):
        super(CorruptedCIFAR10BaselineParamsLR5BatchSize64DomainlessTextProp50, self).__init__()
        self.domainless_text_prop = 0.5

class CorruptedCIFAR10BaselineParamsLR5BatchSize64DomainlessTextProp75(CorruptedCIFAR10BaselineParamsLR5BatchSize64):

    def __init__(self):
        super(CorruptedCIFAR10BaselineParamsLR5BatchSize64DomainlessTextProp75, self).__init__()
        self.domainless_text_prop = 0.75

class CorruptedCIFAR10BaselineParamsLR5BatchSize64DomainlessTextProp25(CorruptedCIFAR10BaselineParamsLR5BatchSize64):

    def __init__(self):
        super(CorruptedCIFAR10BaselineParamsLR5BatchSize64DomainlessTextProp25, self).__init__()
        self.domainless_text_prop = 0.25

#introduce additive loss

class CC10DomainlessTextProp50DisentanglementLambda0_1(CorruptedCIFAR10BaselineParamsLR5BatchSize64DomainlessTextProp50):

    def __init__(self):
        super(CC10DomainlessTextProp50DisentanglementLambda0_1, self).__init__()
        self.do_disentanglement = True
        self.do_disentanglement_ortho = False
        self.disentanglement_modality = 'text'
        self.disentanglement_initialization_batch_size = 1024
        self.disentanglement_batch_size = 128
        self.disentanglement_component_optimizer_type = 'Adam'
        self.disentanglement_component_learning_rate = 1e-3
        self.disentanglement_component_scheduler_type = 'none'
        self.disentanglement_lambda = 0.1

class CC10DomainlessTextProp50DisentanglementLambda1_0(CC10DomainlessTextProp50DisentanglementLambda0_1):

    def __init__(self):
        super(CC10DomainlessTextProp50DisentanglementLambda1_0, self).__init__()
        self.disentanglement_lambda = 1.0

#introduce ortho loss

class CC10DomainlessTextProp50DisentanglementOrthoLambda0_1(CC10DomainlessTextProp50DisentanglementLambda0_1):

    def __init__(self):
        super(CC10DomainlessTextProp50DisentanglementOrthoLambda0_1, self).__init__()
        self.do_disentanglement_ortho = True

class CC10DomainlessTextProp50DisentanglementOrthoLambda1_0(CC10DomainlessTextProp50DisentanglementOrthoLambda0_1):

    def __init__(self):
        super(CC10DomainlessTextProp50DisentanglementOrthoLambda1_0, self).__init__()
        self.disentanglement_lambda = 1.0

#try ramping up lambda (with and without ortho loss)

class CC10DomainlessTextProp50DisentanglementLambda10_0(CC10DomainlessTextProp50DisentanglementLambda0_1):

    def __init__(self):
        super(CC10DomainlessTextProp50DisentanglementLambda10_0, self).__init__()
        self.disentanglement_lambda = 10.0

class CC10DomainlessTextProp50DisentanglementOrthoLambda10_0(CC10DomainlessTextProp50DisentanglementOrthoLambda0_1):

    def __init__(self):
        super(CC10DomainlessTextProp50DisentanglementOrthoLambda10_0, self).__init__()
        self.disentanglement_lambda = 10.0

#even more ramp-up!

class CC10DomainlessTextProp50DisentanglementLambda100_0(CC10DomainlessTextProp50DisentanglementLambda0_1):

    def __init__(self):
        super(CC10DomainlessTextProp50DisentanglementLambda100_0, self).__init__()
        self.disentanglement_lambda = 100.0

class CC10DomainlessTextProp50DisentanglementOrthoLambda100_0(CC10DomainlessTextProp50DisentanglementOrthoLambda0_1):

    def __init__(self):
        super(CC10DomainlessTextProp50DisentanglementOrthoLambda100_0, self).__init__()
        self.disentanglement_lambda = 100.0

#last-ditch effort to make things work by explicitly encouraging image and text to be close to each other


class CC10DomainlessTextProp50DisentanglementOrthoLambda100_0ClosenessLambda1_0(CC10DomainlessTextProp50DisentanglementOrthoLambda100_0):

    def __init__(self):
        super(CC10DomainlessTextProp50DisentanglementOrthoLambda100_0ClosenessLambda1_0, self).__init__()
        self.do_closeness_loss = True
        self.closeness_lambda = 1.0

class CC10DomainlessTextProp50DisentanglementOrthoLambda100_0ClosenessLambda10_0(CC10DomainlessTextProp50DisentanglementOrthoLambda100_0ClosenessLambda1_0):

    def __init__(self):
        super(CC10DomainlessTextProp50DisentanglementOrthoLambda100_0ClosenessLambda10_0, self).__init__()
        self.closeness_lambda = 10.0

class CC10DomainlessTextProp50DisentanglementOrthoLambda100_0ClosenessLambda100_0(CC10DomainlessTextProp50DisentanglementOrthoLambda100_0ClosenessLambda1_0):

    def __init__(self):
        super(CC10DomainlessTextProp50DisentanglementOrthoLambda100_0ClosenessLambda100_0, self).__init__()
        self.closeness_lambda = 100.0

class CC10DomainlessTextProp50DisentanglementOrthoLambda100_0ClosenessLambda0_1(CC10DomainlessTextProp50DisentanglementOrthoLambda100_0ClosenessLambda1_0):

    def __init__(self):
        super(CC10DomainlessTextProp50DisentanglementOrthoLambda100_0ClosenessLambda0_1, self).__init__()
        self.closeness_lambda = 0.1

class CC10DomainlessTextProp50DisentanglementOrthoLambda100_0ClosenessLambda00_1(CC10DomainlessTextProp50DisentanglementOrthoLambda100_0ClosenessLambda1_0):

    def __init__(self):
        super(CC10DomainlessTextProp50DisentanglementOrthoLambda100_0ClosenessLambda00_1, self).__init__()
        self.closeness_lambda = 0.01

def grab_params(params_key):
    return eval(params_key + '()')
