conf_train = {
    # training configuration
    'warmup_epoch': False,
    'num_epochs_warmup_lr': 0,
    'num_epochs_const_lr': 60, # 100
    'num_epochs_const_lr_2': 150, # 300
    'num_epochs': 400,
    'batch_size': 4, # for gradient calculation
    'learning_rate': 0.001/2,
    'learning_rate_2': 0.0001/2,
    'learning_rate_3': 0.00001,
	'momentum': 0,
	'momentum_nesterov': False,
    'weight_decay': 5e-4,
    'clip_grads': 10.0,
    # data configuration
    'noisy_coords': "pair",
    'norm_size': 4 # 8 (as 'batch_size') # computational batch (influences batch normalization)
}