import os, time
from contextlib import redirect_stdout
from datetime import datetime
from functools import partial # for function arguments binding
from typing import List, Union

import torch.nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from ..data_loader import EnzymeClassDataset, Collate_Protein_Batch
from ..model_py import print_n_params
from .. import utils
from .conf_train import conf_train

EPS_ACC = 1e-4 # like PyTorch threshold https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html

def current_milli_time(): return time.time() * 1000.0
def f(): pass

class _Train:
	output_pipe = None # stdout for normal logging
	display_prefix = ""
	_CR = "\r"
	
	def __init__(self,
		model: torch.nn.Module,
		conf_dict: dict,
		feed: type(f),
		dir: str, VERSION, _unused,
		train_set: Union[str, List[str]] = None,
		val_set: str = None,
		dataset_fold: str = None,
		model_name = "" # should be convertible to string
	):
		self.feed = feed
		
		# assure realbatch is smaller than minibatch size
		conf_dict['batch_size'] = conf_dict['batch_size'] or conf_dict['norm_size']
		if 'norm_size' in conf_dict:
			assert conf_dict['norm_size'] <= conf_dict['batch_size'], "norm_size <= batch_size does not hold (%i <= %i)" % (conf_dict['norm_size'], conf_dict['batch_size'])
		
		# set the defaults
		conf_dict = {**conf_train, **conf_dict}
		conf_dict['norm_size'] = min(conf_dict['norm_size'], conf_dict['batch_size']) # if using default norm_size value
		self.conf_dict = conf_dict
		self.conf_dict["DEVICE_DUMP"] = utils.DEVICE_DUMP
		
		# train_set & val_set arguments preprocessing
		if not train_set: # default datasets
			train_set = "validation_pdb_reversed"
			val_set = val_set or "list_pdb_reversed"
		elif not val_set: # train and test are equal, or cross-validation case
			val_set = train_set
			dataset_fold = dataset_fold or "1"
		
		if type(train_set) is str:
			train_set = [train_set]
		if type(val_set) is str:
			val_set = [val_set]
		
		# Create output folder
		dir += "/model%s" % str(model_name)
		if os.path.isdir(dir): # create a new folder with suffix if one exist already
			cur_date = datetime.now()
			dir += cur_date.strftime("_%d-%m-%Y-%H%M%S")
		os.makedirs(dir)
		self.log_folder_path = dir
		
		# Set up logging
		if model_name != "":
			self.display_prefix = str(model_name) + ':'
			self.output_pipe = open(dir + "/log.txt", 'w', 1) # line buffering
			self._CR = "" # don't write carriage return if logging to a file
		lprint = partial(print, file=self.output_pipe) # log printing
		
		torch.set_printoptions(profile="full")

		##################################################################
		self.prot_enc =  model.cuda()
		##################################################################

		print_n_params(model, lprint)
		lprint()
		
		# Tensorboard
		self.writer = SummaryWriter(self.log_folder_path)

		# Dataset fetching
		prot_train_ds = [
			EnzymeClassDataset(
				p_dataset     = train_set, 
				p_random_seed = 33, 
				p_data_aug    = bool(conf_dict['noisy_coords']),
				p_batch_pairs = conf_dict['noisy_coords'] == "pair",
				p_load_data   = False, #True
				p_fold        = dataset_fold,
				p_train_mode  = True
			) for train_set in train_set
		]
		prot_train_ds = ConcatDataset(prot_train_ds)
		lprint("Number of proteins training:", len(prot_train_ds))
		self.prot_training_dataloader = DataLoader(
			prot_train_ds, 
			batch_size  = conf_dict['norm_size'], 
			shuffle     = True, 
			drop_last   = True,
			collate_fn  = Collate_Protein_Batch.collate,
			num_workers = 1 # todo: check if this is a cause of ireproducibility
		)

		prot_val_ds = [
			EnzymeClassDataset(
				p_dataset     = val_set, 
				p_random_seed = 33, 
				p_data_aug    = False,
				p_fold        = dataset_fold,
				#p_load_data=True
			) for val_set in val_set
		]
		prot_val_ds = ConcatDataset(prot_val_ds)
		lprint("Number of proteins validation:", len(prot_val_ds), "(fold %s)" % dataset_fold)
		self.prot_validation_dataloader = DataLoader(
			prot_val_ds, 
			batch_size  = conf_dict['norm_size'], 
			shuffle     = False, 
			collate_fn  = Collate_Protein_Batch.collate,
			num_workers = 1
		)
	
	def cleanup(self):
		if self.output_pipe:
			self.output_pipe.close()
			self.output_pipe = None

	def fit(self):
		conf_dict = self.conf_dict
		prot_enc  = self.prot_enc
		
		lprint = partial(print, file=self.output_pipe) # log printing
		
		# batches per training step
		batches_per_step = conf_dict['batch_size'] / conf_dict['norm_size']

		# Loss function
		# loss_fn = torch.nn.L1Loss()                               # MAE (used here for non-binary classification – is it correct though?)
		loss_fn = torch.nn.BCELoss()                                # binary cross-entropy for a binary classification
		# loss_fn = torch.nn.BCELoss(weight=torch.tensor([1, 6.9])) # class weighting can be applied (positive class is less abundant)

		# Optimizer.
		conf_params_optimizer = [{  'params': prot_enc.parameters()}]
		optimizer = torch.optim.SGD(
			conf_params_optimizer, 
			weight_decay = conf_dict['weight_decay'],
			lr = conf_dict['learning_rate'],
			momentum = conf_dict['momentum'],
			nesterov = conf_dict['momentum_nesterov']
		)

		# Learning rate scheduler
		scheduler_step = lambda metrics: scheduler.step() # throw away metrics as not used in a simple scheduler
		if 'lr_scheduler' in conf_dict:
			scheduler = torch.optim.lr_scheduler.__dict__[conf_dict['lr_scheduler']](optimizer, verbose=True)
			scheduler_step = scheduler.step
		else:
			# DEFAULT fixed scheduler
			lr_muls = []
			for i in range(conf_dict['num_epochs']+1):
				if i < conf_dict['num_epochs_warmup_lr']:
					lr_muls.append((i+1)/conf_dict['num_epochs_warmup_lr'])
				elif i < conf_dict['num_epochs_const_lr']:
					lr_muls.append(1.0)
				elif i < conf_dict['num_epochs_const_lr_2']:
					lr_muls.append(conf_dict['learning_rate_2']/conf_dict['learning_rate'])
				else:
					lr_muls.append(conf_dict['learning_rate_3']/conf_dict['learning_rate'])
			
			lambda_lr = lambda epoch: lr_muls[epoch]
			scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

		# Execute a model with a batch of data and calculates loss and accuracy
		def exe_batch(model, cur_batch):
			
			start_time_date = current_milli_time()			
			cur_labels = cur_batch[-1].cuda()
			cur_weights = cur_batch[-2].cuda()
			
			logits, time_data = self.feed(model, cur_batch[:-2])

			if len(cur_labels) == 1:
				cur_labels = cur_labels[0]
			cur_labels = cur_labels.to(torch.float32).reshape(logits.shape)

			# Set weights per an example
			loss_fn.weight = cur_weights.reshape(-1,1)
			loss = loss_fn(logits, cur_labels)
			
			# Compute (incomplete) confusion matrix
			preds = torch.round(logits).to(torch.int)
			cur_labels = cur_labels.to(torch.int)
			metrics = dict(
				hits = (preds == cur_labels).to(torch.int).sum().item(),
				TP   = (torch.bitwise_and(preds, cur_labels)).to(torch.int).sum().item(),
				FP   = (torch.bitwise_and(preds, torch.bitwise_not(cur_labels))).to(torch.int).sum().item(),
				N    = len(logits)
			)
			
			return logits, loss, metrics, time_data

		# ---------------------------------------- TRAINING PROCESS ----------------------------------------
		prot_enc = prot_enc.eval()
		best_loss = 100.0
		best_loss_ep = 0
		best_acc = 0.0
		best_acc_ep = 0
		try:
			for cur_epoch in range(conf_dict['num_epochs']):

				lprint()
				lprint("### Epoch", cur_epoch)

				start_time_epoch = time.time()
				
				stage_metrs = {'train': None, 'val': None}
				for stage in ("train", "val"):
					
					if stage == "train":
					# ---------------------------------------- LEARNING STAGE ----------------------------------------
						prot_enc = prot_enc.train() # Start training
						dataloader = self.prot_training_dataloader
					else:
					# ---------------------------------------- VALIDATION STAGE ----------------------------------------
						prot_enc = prot_enc.eval() # Switch to validation mode
						dataloader = self.prot_validation_dataloader
				
					iterator = 0
					self.loss = 0.0
					self.acc = 0.0
					metrs = dict(hits = 0, TP = 0, FP = 0, N = 0)
					# Iterate over the training data batch by batch
					for iterator, cur_batch in enumerate(dataloader):
						start_time = current_milli_time()
						
						logits, loss_val, cur_metrics, time_data =                                                        \
							exe_batch(prot_enc, cur_batch)

						# ---------------------------------------- LEARNING ----------------------------------------
						if stage == "train":
							loss_val.backward()
							
							if conf_dict['warmup_epoch'] and cur_epoch == 0:
								# LR = initial_lr * i / steps
								optimizer.param_groups[0]['lr'] = conf_dict['learning_rate'] * (iterator+1) / len(dataloader) # optimizer.param_groups[0]['initial_lr'] # this works only with LambdaLR scheduler

							if conf_dict['clip_grads'] > 0.0:
								torch.nn.utils.clip_grad_norm_(prot_enc.parameters(), conf_dict['clip_grads'])
							
							# perform training step (weights update)
							# artificially increasing batch size to limit memory usage
							if iterator % batches_per_step == 0: # note: int % 0.x -> 0
								optimizer.step()
								optimizer.zero_grad()
						# ---------------------------------------- LEARNING END ----------------------------------------

						metrs = {k: metrs[k] + cur_metrics[k] for k in cur_metrics}
						self.acc = metrs['hits'] / metrs['N'] * 100
						self.loss += (loss_val.item() - self.loss)/(iterator+1)

						end_time = current_milli_time()

						if iterator%10 == 0:
							print("\r{:s} ({:4d}/{:4d}) Loss: {:.6f} Acc: {:.2f} (TD: {:.4f}  TT: {:.4f}   LR:{:#.2g})           ".format(
								self.display_prefix, # to distiunguish models
								iterator, 
								len(dataloader), 
								self.loss, self.acc,
								time_data,
								end_time - start_time - time_data,
								optimizer.param_groups[0]['lr']), # keep exactly (#) 2 significant (g) decimal places
							end="")

					# Calculate statistics
					metrs['TN'] = metrs['hits'] - metrs['TP']
					miss = metrs['N'] - metrs['hits']
					FN = miss - metrs['FP']
					sensitivity = metrs['TP'] / (metrs['TP'] + FN)
					specificity = metrs['TN'] / (metrs['TN'] + metrs['FP'])
					BA = (sensitivity + specificity) / 2 * 100
					for m in ('miss', 'sensitivity', 'specificity', 'BA'):
						metrs[m] = eval(m)
					metrs['acc'] = self.acc
					metrs['loss'] = self.loss
					
					stage_metrs[stage] = metrs

					# Save statistics
					self.writer.add_scalars(stage + '_perf', {
						'accuracy': self.acc,
						'BA': BA
					}, cur_epoch)
					self.writer.add_scalar(stage + '_loss', self.loss, cur_epoch)

					lprint(self._CR + "{:s} Loss: {:.4f} Accuracy: {:.2f} BA: {:.2f}                                       ".format(
						"Training  " if stage == "train" else "Validation",
						self.loss, self.acc, BA
					))

				# ---------------------------------------- ONE EPOCH END ----------------------------------------
				self.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], cur_epoch)
				
				# Save current model
				conf_dict['state_dict_prot_enc'] = prot_enc.state_dict()
				torch.save(conf_dict, self.log_folder_path+"/model_saved.pth")
				# Save the so-far-best models
				if self.loss < best_loss:
					best_loss = self.loss
					best_loss_ep = cur_epoch
					torch.save(conf_dict, self.log_folder_path+"/best_model_loss.pth")
				if self.acc > best_acc + EPS_ACC :
					best_acc = self.acc
					best_acc_ep = cur_epoch
					torch.save(conf_dict, self.log_folder_path+"/best_model_acc.pth")

				end_time_epoch = time.time()
				lprint("End epoch {:.4f}".format(end_time_epoch-start_time_epoch))

				# Prepare for next epoch
				scheduler_step(self.loss)

				yield stage_metrs
				# EARLY STOPPING if no improving last n epochs
				if 'lr_scheduler' in conf_dict and\
				  cur_epoch - best_loss_ep > 15 and\
				  cur_epoch - best_acc_ep  > 15:
					break
				
		finally: # when exit/fails/terminates
			lprint()
			lprint()
			lprint("Best accuracy: {:.2f} (epoch {:d})                                   ".format(best_acc, best_acc_ep))
			lprint("Best loss: {:.2f} (epoch {:d})                                       ".format(best_loss, best_loss_ep))
			lprint()
			self.cleanup() # todo: is it needed?


# functional wrapper for the Train class with a self-execution
def Train(*args, **kwargs):
	training = _Train(*args, **kwargs).fit()
	
	try: # allow to abort training this model by a keaboard interrupt
		for model in training:
			pass
	except KeyboardInterrupt:
		pass
	
	return model


# Train ensemble of models
class TrainMultiple:
	
	def __init__(
		self,
		models: list,
		conf_dict: dict,
		feed: type(f),
		dir: str,
		*args,
		name_prefix: str = "", # cannot be assigned positionaly (backward compatibility)
		**kwargs
	):
		
		writer = SummaryWriter(dir + "/metrics")
		
		with open(dir + "/device.log", 'w') as log:
			utils.	dump_device(log)
		
		trainings = [_Train(
			model,
			conf_dict,
			feed,
			dir,
			model_name = name_prefix+str(i),
			dataset_fold=str(i+1),
			*args, **kwargs
		).fit() for i, model in (
				(i, model) for i, model in enumerate(models)
			if model is not None) # skip Nones
		]
		# todo: maintain reproducibility, setting these 3 is not enough (maybe dataset loader workers are curlprit)
		# torch.manual_seed(42)
		# np.random.seed(0)
		# random.seed(0)
		running_results = [{}] * 5
		for e in range(conf_dict['num_epochs']):
			print()
			print("### Epoch", e)
				
			running_results = [next(metrs, running_results[i]) for i,metrs in enumerate(trainings)]
			
			metrs = {'train': {}, 'val': {}}
			# Make statistics across the ensemble
			for stage in running_results[0].keys():
				print("\rTraining  " if stage == "train" else "Validation", end='')
				for var in ('loss', 'acc', 'BA', ):
					values = metrs[stage][var] = [m[stage][var] for m in running_results]
					mean = sum(values) / len(models)
					print(" {:s}: {:.4g} ({:.4g}-{:.4g})".format(var.upper(), mean, min(values), max(values)), end='')
					
					writer.add_scalars(stage + '_' + var, {
							'mean': mean,
							'min': min(values),
							'max': max(values)
						}, e
					)
				print()
		
		writer.close()