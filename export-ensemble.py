# example use: python export-ensemble.py ./experiments/train-v21_2-on-LGK-dupl/model? --test [--saveas <PATH>]
import sys

import torch
from torch.utils.data import DataLoader

from code.data_preprocessing import process_pdb, mutate_seq
from code.model_py.v21_2 import Ensemble
from code.model_py import print_n_params
from code.data_loader import create_datapoint, collate_batch

MODEL_EXPORT = 'model_weights__.pth'


if __name__ == "__main__":
	should_test = '--test' in sys.argv
	models = sys.argv[1:]
	if should_test:
		models.remove('--test')
	export_path = 'model_weights__.pth'
	if "--saveas" in models:
		idx = models.index("--saveas")
		export_path = models[idx + 1]
		del models[idx:idx+2]
	models = [path + "/best_model_acc.pth" for path in models]
	
	from code.data_preprocessing.model import get_PDB

	# Test structure
	if True: # dummy example
		pdb_code = "2ci2"
		chain = 'I'
		length = 5 # length of protein to test
		mutation = None
	else:
		chain = 'A'
		length = None
		# mutation = (['W'], [140-5], ['F'])
		# pdb_code, mutation = ("1col", (['W'], [140-5], ['C']))
		pdb_code, mutation = ("1z0q", (['F'], [19], ['A'])) # 1z0q.FA19A.A

	print(pdb_code, mutation)
	# ############################## Structure preprocessing
	pdb_file = get_PDB(pdb_code)

	orig_seq, orig_coords = process_pdb(pdb_file, chain)
	# orig_seq, orig_coords = orig_seq[:length], orig_coords[:length]

	protein = protein2 = create_datapoint(
		pdb_code,
		orig_seq,
		orig_coords
	)
	if mutation:
		protein2 = create_datapoint(
			pdb_code,
			mutate_seq(orig_seq, *mutation),
			orig_coords
		)
	batch = collate_batch([protein, protein2])[1:-2] # and drop name, weights and labels
	# batch = collate_batch([protein, protein2, protein, protein2, protein2, protein])[1:-2] # and drop name, weights and labels
	
	# print(orig_seq, orig_coords) # remove the chain name from the input
	# print(batch)
	print()
	
	# ############################## Prepare model
	model = Ensemble(models)

	torch.no_grad()
	model.eval()
	print_n_params(model)

	print(model(*batch))
	
	# ############################## Execution
	
	if should_test: # tests the exported model
		conf_dict = torch.load(export_path)
		# print(conf_dict.keys())
		saved_model = Ensemble(5)
		# model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
		saved_model.load_state_dict(conf_dict)
		saved_model.eval()
		print(saved_model(*batch))
		
		# TorchScript version (does not work)
		# torchscript = torch.load('model_scripted.pt')
		# print(torchscript(*batch))
	else:
		torch.save(model.state_dict(), export_path)
		
		# TorchScript version
		# may work with newer PYG version (2.5+)
		# torchscript = torch.jit.script(model.models[0]) # does not work: object has no attribute or method 'edge_index', related to Data class of PYG, problem in PYG v2.6 too
		# torchscript = torch.jit.trace(model, batch) # using jit.trace instead, which requires an example input
		# torchscript.save('model_scripted.pt')

