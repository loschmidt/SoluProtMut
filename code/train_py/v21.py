import os, time
from contextlib import redirect_stdout
from datetime import datetime

import torch.nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..data_loader import EnzymeClassDataset, Collate_Protein_Batch
from ..model_py import print_n_params

def current_milli_time(): return time.time() * 1000.0
def f(): pass

class Train:

	def __init__(self,
		model: torch.nn.Module,
		conf_dict: dict,
		feed: type(f),
		dir: str, VERSION, VERSION_DESC,
		train_set: str = "validation_pdb_reversed",
		val_set: str = "list_pdb_reversed"
	):

		torch.set_printoptions(profile="full")

		##################################################################
		prot_enc =  model.cuda()
		##################################################################

		print_n_params(prot_enc)
		print()

		# Log folder
		cur_date = datetime.now()
		dt_string = cur_date.strftime("__%d_%m_%Y__%H_%M_%S")
		log_folder_path = "runs/log_v%s" % VERSION + dt_string
		# os.makedirs(log_folder_path)
		log_folder_path = dir
		
		# online logging
		# LOG = LOG()

		# Tensorboard.
		writer = SummaryWriter(log_folder_path)

		# Training dataset
		prot_train_ds = EnzymeClassDataset(
			p_dataset     = train_set, 
			p_random_seed = 33, 
			p_data_aug    = bool(conf_dict['noisy_coords']),
			p_batch_pairs = conf_dict['noisy_coords'] == "pair",
			p_load_data   = False, #True,
			#p_min_seq_remove = conf_dict['min_mask_da'], # these are not in the data_loader.py
			#p_max_seq_remove = conf_dict['max_mask_da'],
			#p_sub_sec_aug = conf_dict['mask_seq_aug']
		)
		print("Number of proteins training:", len(prot_train_ds))
		prot_training_dataloader = DataLoader(
			prot_train_ds, 
			batch_size  = conf_dict['norm_size'], 
			shuffle     = True, 
			drop_last   = True,
			collate_fn  = Collate_Protein_Batch.collate,
			num_workers = 1
		)

		prot_val_ds = EnzymeClassDataset(
			p_dataset     = val_set, 
			p_random_seed = 33, 
			p_data_aug    = False,
			#p_load_data=True
		)
		print("Number of proteins validation:", len(prot_val_ds))
		prot_validation_dataloader = DataLoader(
			prot_val_ds, 
			batch_size  = conf_dict['norm_size'], 
			shuffle     = False, 
			collate_fn  = Collate_Protein_Batch.collate,
			num_workers = 1
		)

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
			momentum = 0 # TODO: conf_dict
		)

		# Learning rate scheduler
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
			# global deltaF
			nonlocal feed
			
			start_time_date = current_milli_time()			
			cur_labels = cur_batch[-1].cuda()
			cur_weights = cur_batch[-2].cuda()
			
			logits, time_data = feed(model, cur_batch[:-2])

			if len(cur_labels) == 1:
				cur_labels = cur_labels[0]
			cur_labels = cur_labels.to(torch.float32).reshape(logits.shape)
			
			# delta = deltaF.predict(res_wt, res_mut)
			# delta = res_mut - res_wt
			# desired = torch.ones(len(delta)).reshape(-1, 1).cuda()
			
			# weights = torch.tensor([6.9 if l == 1 else 1 for l in cur_labels])
			# loss = loss_fn(logits, cur_labels, weights)
			loss_fn.weight = cur_weights.reshape(-1,1)
			loss = loss_fn(logits, cur_labels)
			
			# Compute (incomplete) confusion matrix
			preds = torch.round(logits).to(torch.int)
			cur_labels = cur_labels.to(torch.int)
			metrics = dict(
				hits = (preds == cur_labels).to(torch.int).sum().item(),
				TP   = (torch.bitwise_and(preds, cur_labels)).to(torch.int).sum().item(),
				FP   = (torch.bitwise_and(preds, torch.bitwise_not(cur_labels))).to(torch.int).sum().item()
			)
			
			return logits, loss, metrics, time_data

		# Training process
		prot_enc = prot_enc.eval()
		best_loss = 100.0
		best_loss_ep = 0
		best_acc = 0.0
		best_acc_ep = 0
		try:
			for cur_epoch in range(conf_dict['num_epochs']):

				print()
				print("### Epoch", cur_epoch)
				print()

				start_time_epoch = time.time()


				# ---------------------------------------- LEARNING STAGE ----------------------------------------
				# Start training
				prot_enc = prot_enc.train()
				
				iterator = 0
				accum_loss = 0.0
				accum_acc = 0.0
				metrs = dict(hits = 0, TP = 0, FP = 0)
				N = 0
				# Iterate over the training data batch by batch
				for iterator, cur_batch in enumerate(prot_training_dataloader):
					start_time = current_milli_time()
					
					logits, loss_val, cur_metrics, time_data =                                                         \
						exe_batch(prot_enc, cur_batch)

					loss_val.backward()
					
					if conf_dict['warmup_epoch'] and cur_epoch == 0:
						# LR = initial_lr * i / steps
						optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['initial_lr'] * (iterator+1) / len(prot_training_dataloader)

					if conf_dict['clip_grads'] > 0.0:
						torch.nn.utils.clip_grad_norm_(prot_enc.parameters(), conf_dict['clip_grads'])
					# perform training step (weights update)
					if iterator % batches_per_step == 0: # artificially increasing batch size to limit memory usage
					# loss_val /= batches_per_step
						optimizer.step()
						optimizer.zero_grad()


					metrs = {k: metrs[k] + cur_metrics[k] for k in cur_metrics}
					TN = metrs['hits'] - metrs['TP']
					N += len(logits)
					accum_acc = metrs['hits'] / N * 100
					accum_loss += (loss_val.item() - accum_loss)/(iterator+1)

					end_time = current_milli_time()

					if iterator%10 == 0:
						print("\r ({:4d}/{:4d}) Loss: {:.6f} Acc: {:.2f} (TD: {:.4f}  TT: {:.4f}   LR:{:#.2g})           ".format(
							iterator, 
							len(prot_training_dataloader), 
							accum_loss, accum_acc,
							time_data,
							end_time - start_time - time_data,
							optimizer.param_groups[0]['lr']), # keep exactly (#) 2 significant (g) decimal places
						end="")
					
					# LOG.iter()

				miss = N - metrs['hits']
				FN = miss - metrs['FP']
				sensitivity = metrs['TP'] / (metrs['TP'] + FN)
				specificity = TN / (TN + metrs['FP'])
				BA = (sensitivity + specificity) / 2 * 100

				writer.add_scalars('tr_perf', {
					'accuracy': accum_acc,
					'BA': BA
				}, cur_epoch)
				writer.add_scalar('tr_loss', accum_loss, cur_epoch)

				print("\rTraining Loss:   {:.4f} Accuracy: {:.2f} BA: {:.2f}                                       ".format(accum_loss, accum_acc, BA))

				# ---------------------------------------- VALIDATION STAGE ----------------------------------------
				# Switch to validation mode
				prot_enc = prot_enc.eval()

				# Iterate over the validation data.
				iterator = 0
				accum_loss = 0.0
				accum_acc = 0.0
				metrs = dict(hits = 0, TP = 0, FP = 0)
				N = 0
				for iterator, cur_batch in enumerate(prot_validation_dataloader):
					start_time = current_milli_time()
					logits, loss_val, cur_metrics, time_data =                                                         \
						exe_batch(prot_enc, cur_batch)

					metrs = {k: metrs[k] + cur_metrics[k] for k in cur_metrics}
					TN = metrs['hits'] - metrs['TP']
					N += len(logits)
					accum_acc = metrs['hits'] / N * 100
					accum_loss += (loss_val.item() - accum_loss)/(iterator+1)

					end_time = current_milli_time()

					if iterator%10==0:
						print("\r ({:4d}/{:4d}) Loss: {:.6f} Acc: {:.2f} (TD: {:.4f}  TT: {:.4f})           ".format(
							iterator, 
							len(prot_validation_dataloader), 
							accum_loss, accum_acc,
							time_data,
							end_time - start_time - time_data), end="")

				miss = N - metrs['hits']
				FN = miss - metrs['FP']
				sensitivity = metrs['TP'] / (metrs['TP'] + FN)
				specificity = TN / (TN + metrs['FP'])
				BA = (sensitivity + specificity) / 2 * 100

				writer.add_scalars('val_perf', {
					'accuracy': accum_acc,
					'BA': BA
				}, cur_epoch)
				writer.add_scalar('val_loss', accum_loss, cur_epoch)

				print("\rValidation Loss: {:.4f} Accuracy: {:.2f} BA: {:.2f}                                       ".format(accum_loss, accum_acc, BA))

				# save current model
				conf_dict['state_dict_prot_enc'] = prot_enc.state_dict()
				torch.save(conf_dict, log_folder_path+"/model_saved.pth")

				# save the-so-far-best models
				if accum_loss < best_loss:
					best_loss = accum_loss
					best_loss_ep = cur_epoch
					torch.save(conf_dict, log_folder_path+"/best_model_loss.pth")
				if accum_acc > best_acc:
					best_acc = accum_acc
					best_acc_ep = cur_epoch
					torch.save(conf_dict, log_folder_path+"/best_model_acc.pth")

				end_time_epoch = time.time()
				print("End epoch {:.4f}".format(end_time_epoch-start_time_epoch))

				scheduler.step()
		finally: # when exit/fails/terminates
			print()
			print()
			print("Best accuracy: {:.2f} (epoch {:d})                                   ".format(best_acc, best_acc_ep))
			print("Best loss: {:.2f} (epoch {:d})                                       ".format(best_loss, best_loss_ep))
			print()

