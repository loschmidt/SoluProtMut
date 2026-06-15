# evaluates all the trained models specified in MODELS on the datasets in DATASETS 
import glob
import pickle
from os.path import dirname

import pandas, torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from torch.utils.data import DataLoader

from code.data_loader import EnzymeClassDataset, Collate_Protein_Batch
from code.model_py.v21_1 import MutPred, PlaNNet, feed, VERSION, VERSION_DESC


def restore_model(model, type="acc"): # can be loss as well
	conf_dict = torch.load("%s/best_model_%s.pth" % (model, type))
	model = MutPred(PlaNNet(conf_dict['gl_pooling'], bool(conf_dict['embed_aa']), conf_dict['embed_aa'] == 'learn'))
	model.load_state_dict(conf_dict["state_dict_prot_enc"], strict=False) # embedding values may be missing
	model.cuda().eval()
	return model

def eval_model(model, dataset):
	batch_size = 16
	val_ds = EnzymeClassDataset(
		p_dataset     = dataset, 
		p_data_aug    = False
	)
	dataloader = DataLoader(
		val_ds, 
		batch_size  = batch_size, 
		shuffle     = False, 
		collate_fn  = Collate_Protein_Batch.collate,
		num_workers = 1 # slower when higher than 1
	)

	# iterate over all the data
	partial = []
	rows = []
	for cur_batch in dataloader:
		rows += cur_batch[0][::-2]
		
		logits, _ = feed(model, cur_batch[:-2]) # feed without labels and weights

		# Compute (incomplete) confusion matrix
		# cur_weights = cur_batch[-2]
		cur_labels = cur_batch[-1].to(torch.int).cuda()
		logits = logits.reshape(cur_labels.shape)
		preds = torch.round(logits).to(torch.int)
		hits =  preds == cur_labels
		
		partial.append(torch.stack((cur_labels, logits, hits)).detach().cpu().to(torch.float))

	results = torch.cat(partial, 1).transpose(0,1)
	results = pandas.DataFrame(results, columns=('true', 'pred', 'hit')).astype({"true": "int", "hit": "bool"})
	results.insert(0, "example", rows)

	return classification_report(results.true, results.pred.round(), output_dict=True), results

# models trained with splits done (i) purely randomly or (ii) position-wise
MODELS = ["""
train-on-DMS/model*
train-on-LGK-dupl/model*
train-on-LGK-GFP/model*
train-on-PKS/model*
train-on-TEM-v22/model*
train-on-DMS-leave-LGK/model*
train-on-DMS-leave-PKS/model*
train-on-DMS-leave-TEM/model*
train-on-DMS-leave-TEM/model*
fine-tuned-TEM/model
fine-tuned-PKS/model
fine-tuned-DMS/model
""".split(), """
train-bypos-on-LGK-dupl/model*
train-bypos-on-PKS/model*
train-bypos-on-TEM/model*
""".split()]

DATASETS = ["""
LGK_GFP_bymut.v1.test
LGK_dupl.v1.test
PKS_bymut.v1.test
TEM_bymut.test
nonDMS.test
LGK_GFP_bymut.v1.train
LGK_dupl_bymut.v1.train
PKS_bymut.v1.train
TEM_bymut.train
nonDMS.train
""".split(), """
LGK_dupl.v1.test
TEM_bypos.v1.test
PKS_bypos.v1.test
nonDMS.test
LGK_dupl_bypos.v1.train
TEM_bypos.v1.train
PKS_bypos.v1.train
nonDMS.train
""".split()]

BEST_MODEL = "acc"; RESULTS_F = "results.p"
# BEST_MODEL = "loss"; RESULTS_F = "results-loss.p"
try:
	with open(RESULTS_F, "rb") as f:
		results = pickle.load(f)
except IOError:
	results = {}

for models, datasets in zip(MODELS, DATASETS):
	for path in models:
		model_res = results.get(path, {})
			
		for dataset in datasets:
			if dataset in model_res:
				continue
			model_res[dataset] = {}
			print("Evaluating %s on %s" % (path, dataset))
			
			raw_results = None
			# evaluate model by model in the ensemble
			path_ensemble = "experiments/" + path
			for model in glob.glob(path_ensemble):
				model_name = model[len(dirname(path_ensemble))+1:]
				
				nn = restore_model(model, BEST_MODEL)
				model_res[dataset][model_name], eval_results = eval_model(nn, dataset)
				if raw_results is None:
					raw_results = eval_results[['example', 'true']]
				raw_results[model_name] = eval_results['pred']
			
			# ensemble prediction
			raw_results.insert(2, 'pred_maj', raw_results.iloc[:, 2:].median(axis=1))
			raw_results.insert(2, 'pred_avg', raw_results.iloc[:, 2:].mean(axis=1))
			
			# store the prediction results
			results_filename = "results-%s.csv" % dataset
			raw_results.to_csv(dirname(path_ensemble) + "/" + results_filename, index=False)
		
		results[path] = model_res
		pickle.dump(results, open(RESULTS_F, "wb"))