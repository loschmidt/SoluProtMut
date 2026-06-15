import os, glob
from pathlib import Path

import torch

from code.model_py.v21_2 import MutPred, PlaNNet, feed, VERSION, VERSION_DESC
from code.train_py.v22_1 import Train

# because training with normal momentum does not seem to work

conf_default = {
	# model configuration
	'embed_aa'             : True, # True (VHSE) | False | 'learn'
	'gl_pool'              : 'avg', # both|avg
	'L1_features'          : 128, # 128|256
	'cl_features'          : 1024, # classifier hidden neurons count
	'conv_features'        : 8, # 8 initially, 32 in the paper
	# training configuration
	'warmup_epoch'         : False,
	'num_epochs'           : 200,
	'batch_size'           : 4, # for gradient calculation
	'lr_scheduler'         : "ReduceLROnPlateau", # factor = 0.25
	'learning_rate'        : 0.001,
	'momentum'             : 0.01,
	'momentum_nesterov'    : True,
	'weight_decay'         : 5e-4,
	'clip_grads'           : 10.0,
	# data configuration
	'noisy_coords'         : "pair",
	# 'norm_size'            : 4 # 8 (as 'batch_size') # computational batch (influences batch normalization)
}

sweeps = {
	'learning_rate': {
		'shorthand': 'lr',
		'values': [0.01, 0.001]
	}, 'momentum_nesterov': {
		'shorthand': 'm(n)',
		'values': []
	}, 'momentum': {
		'shorthand': 'm',
		# 'best_value': 0.95,
		'values': [0.01, 0.7, 0.9, 0.95, 0.98] # a, inspired by arXiv:1803.09820 (0.97 and 0.99 merged into 0.98)
		# assuming plateau (still gradient), the effective learning rate increase 3×, 10×, 20×, 50× respectively
		# i.e. such v_n that    v_t = v_t+1 = a.v_t + G, for a=0.7 and G=1: 1 / (1 - 0.7) = 3
	}, 'weight_decay': {
		'shorthand': 'WD',
		'values': [1.5e-3, 5e-4, 1e-4]
	}, 'embed_aa': {
		'shorthand': 'eAA',
		'values': [False, 'learn']
	}, 'batch_size': {
		'shorthand': 'mBS',
		'values': [2, 4, 8, 16, 32, 48, 64]
	}, 'norm_size': {
		'shorthand': 'RB',
		'values': [2, 8, 10, 12, 16, 24, 32, 40] # 4 is default
	}, 'gl_pool': {
		'shorthand': 'p',
		'values': ['avg', 'both']
	}, 'warmup_epoch': {
		'shorthand': 'WuE',
		# 'best_value': True,
		'values': [False, True]
	}, 'L1_features': {
		'shorthand': '1L',
		'values': [64, 96, 192, 256]
	}, 'cl_features': {
		'shorthand': 'CL',
		# 'best_value': 128,
		'values': [128, 256, 512]
	}
}

def stringify(var):
	if type(var) == bool:
		var = int(var)
	
	return str(var)

def print_params(conf: dict):
	params = ""
	for hname in sweeps.keys():
		if hname in conf:
			hval = conf[hname]
			params += ",%s=%s" % (sweeps[hname]["shorthand"], stringify(hval))
	return params

dir = os.path.dirname(__file__)

def sweep_parameters(dataset):
	from inspect import currentframe, getframeinfo, stack
	
	file_name = getframeinfo(stack()[1][0]).filename # get caller's filename
	model_prefix = Path(file_name).stem
	
	# set the best params & skip the defaults
	conf_best = {}
	for hname, hparam in sweeps.items():
		hopt = hparam.get('best_value')
		if hopt is not None:
			conf_best[hname] = hopt
	
	# sweepable params
	for hname in sweeps.keys():
		hparam = sweeps[hname]
		
		ef_params = list(hparam["values"])
		def_val = conf_default.get(hname)
		if def_val in ef_params:
			ef_params.remove(def_val)
		config = dict(conf_best)
		
		for hval in ef_params:
			config[hname] = hval
			
			model_name = model_prefix + print_params(config)
			model_dir = dir + "/model" + model_name
			if Path(model_dir).is_dir(): # skip already executed options
				print(model_name + " skipped")
				continue
			
			config = {**conf_default, **config}
		
			model = MutPred(PlaNNet(
					embed_learn = config['embed_aa'] == 'learn',
					**config
				))
			
			try:
				Train(model, config, feed, dir, VERSION, VERSION_DESC, dataset, model_name=model_name)
			except AssertionError:
				print(model_name + " failed")
