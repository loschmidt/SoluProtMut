import torch

from code.data_preprocessing import process_pdb, mutate_seq
from code.data_loader import create_datapoint, collate_batch


class EnsemblePredictor:
	
	def __init__(self,
		weights: str = "model_weights.pth", # trained model
		version: str = None,                # model version: '21_1'|...
	):
		conf_dict = torch.load(weights)
		
		if version is None: # default model
			from code.model_py import Ensemble
		elif version == '21_2':
			from code.model_py.v21_2 import Ensemble
		elif version == '21_1':
			from code.model_py.v21_1 import Ensemble
		else:
			raise 'Non-existing version!'
		
		pred_model = Ensemble(5)
		pred_model.load_state_dict(conf_dict)
		pred_model.eval()
		
		self._model = pred_model

	def predict_change(self,
		PDB_path: str,
		chain: str,
		aa_from: list,
		locs: list,
		aa_to: list
	):
		assert len(aa_from) == len(locs) == len(aa_to)
		
		print("Processing structure...")
		orig_seq, coords = process_pdb(PDB_path, chain)
		mut_seq = mutate_seq(orig_seq, aa_from, locs, aa_to)
		
		print("Calculating prediction...")
		protein = create_datapoint(
			'wt',
			orig_seq,
			coords
		)
		mutant = create_datapoint(
			'mut',
			mut_seq,
			coords
		)
		
		data = collate_batch([protein, mutant])[:-2]
		prediction = float(self._model.feed(data)[0])
		prediction_cl = 'N' if round(prediction, 2) == 0.5 else\
					'+' if prediction > 0.5 else '-'
		
		return prediction_cl, prediction