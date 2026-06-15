import os
import glob
import torch

from code.model_py.v21_1 import MutPred, PlaNNet, feed, VERSION, VERSION_DESC
from code.train_py.v22 import Train, TrainMultiple

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
    'batch_size'           : 8, # for gradient calculation
    'lr_scheduler'         : "ReduceLROnPlateau", # flexible scheduler!
    'learning_rate'        : 0.001/2, 
    'weight_decay'         : 5e-4,
    'clip_grads'           : 10.0,
    # data configuration
    'noisy_coords'         : "pair",
    'norm_size'            : 8 # 8 (as 'batch_size') # computational batch (influences batch normalization)
}

def restore_model(model, type="acc"): # let's go with loss
	conf_dict = torch.load("%s/best_model_%s.pth" % (model, type))
	model = MutPred(PlaNNet(conf_dict['gl_pooling'], bool(conf_dict['embed_aa']), conf_dict['embed_aa'] == 'learn'))
	model.load_state_dict(conf_dict["state_dict_prot_enc"], strict=False) # embedding values may be missing
	model.cuda().eval()
	
	# freeze model weights
	for param in model.parameters():
		param.requires_grad = False
	model.base_nn.classifier.requires_grad_()
	
	return model

models = []
for model in glob.glob("experiments/train-on-DMS/model*"):
	models.append(restore_model(model, "loss"))

dir = os.path.dirname(__file__)
TrainMultiple(models, conf_dict, feed, dir, VERSION, VERSION_DESC, "nonDMS.train")