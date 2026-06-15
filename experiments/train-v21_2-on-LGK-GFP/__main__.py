import os

from code.model_py.v21_2 import MutPred, PlaNNet, feed, VERSION, VERSION_DESC
from code.train_py.v22_1 import TrainMultiple

conf_dict = {
    # model configuration
    'embed_aa'             : True, # True (VHSE) | False | 'learn'
    'gl_pooling'           : 'avg', # both|avg
    '1stL_features'        : 128, # 128|256
    'conv_features'        : 8, # 8 initially, 32 in the paper
    # training configuration
    'warmup_epoch'         : False,
    'num_epochs_warmup_lr' : 0,
    'num_epochs_const_lr'  : 60,
    'num_epochs_const_lr_2': 120,
    'num_epochs'           : 200,
    'batch_size'           : 4, # for gradient calculation
    'learning_rate'        : 0.001/2,
    'learning_rate_2'      : 0.0001/2,
    'learning_rate_3'      : 0.00001,
    'weight_decay'         : 5e-4,
    'clip_grads'           : 10.0,
    # data configuration
    'noisy_coords'         : "pair",
    'norm_size'            : 4 # 8 (as 'batch_size') # computational batch (influences batch normalization)
}

n_models = 5
models = [
	MutPred(PlaNNet(conf_dict['gl_pooling'], bool(conf_dict['embed_aa']), conf_dict['embed_aa'] == 'learn')) for _ in range(n_models)
]
dir = os.path.dirname(__file__)
TrainMultiple(models, conf_dict, feed, dir, VERSION, VERSION_DESC, "LGK_GFP_bymut.v1.train")