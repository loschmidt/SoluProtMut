import numpy as np
import torch
from code.data_preprocessing.model import get_PDB, FoldX
from code.data_preprocessing.process import process_pdb

# Test structure
pdb_code = "2ci2"
chain = 'I'
length = 5 # length of protein to test

# ############################## PART 1: Structure preprocessing
pdb_file = get_PDB(pdb_code)
orig, mut = FoldX(pdb_file)

orig_seq, orig_coords = process_pdb(orig, chain)
orig_seq, orig_coords = orig_seq[:length], orig_coords[:length]


# ############################## PART 2: Model inference
from code.model_py.v21_1 import conf_dict, MutPred, PlaNNet, feed
# from code.model_py import conf_dict, MutPred, PlaNNet, feed
from code.model_py import print_n_params
from code.data_loader import create_datapoint, collate_batch

model = PlaNNet(conf_dict['gl_pooling'])
model.eval()
print_n_params(model)
print()

protein = create_datapoint(
	pdb_code,
	orig_seq,
	orig_coords
)
batch = collate_batch([protein, protein])[:-2] # and drop weights and labels
print(batch)

print(feed(
	model,
	batch
))

print(feed(
	model,
	batch
))