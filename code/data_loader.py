import os
import math

import numpy as np

from Bio.PDB.Polypeptide import one_to_index

import torch
from torch.utils.data import Dataset

class Collate_Protein_Batch():

	@staticmethod
	def collate(p_batch):
		batch_names    = []
		batch_aas      = []
		batch_coords   = []
		batch_seq_pos  = []
		batch_axes     = []
		batch_instance = []
		batch_labels   = []
		batch_weights  = []
		
		cur_iter = 0
		for protA, protB, label, w in p_batch:
			for chain in (protA, protB):
				if chain:
					batch_names.append(chain[0])
					batch_aas.append(chain[1])
					batch_coords.append(chain[2])
					batch_seq_pos.append(chain[3])
					batch_axes.append(chain[4])
					batch_instance.append(np.ones_like(chain[1])*cur_iter)
					cur_iter += 1
			batch_labels.append(label)
			batch_weights.append(w)
		
		batch_labels = list(filter(lambda l: l is not None, batch_labels))
		
		return batch_names,\
			torch.as_tensor(np.concatenate(batch_aas, axis=0)),\
			torch.as_tensor(np.concatenate(batch_coords, axis=0)),\
			torch.as_tensor(np.concatenate(batch_seq_pos, axis=0)),\
			torch.as_tensor(np.concatenate(batch_axes, axis=0)),\
			torch.as_tensor(np.concatenate(batch_instance, axis=0)).to(torch.int32),\
			torch.as_tensor(batch_weights),\
			torch.as_tensor(batch_labels)


# AA Letter to id
AA1 = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_ID = {}
for i in range(0, 21):
	AA_TO_ID[AA1[i]] = i

def create_datapoint(pdb_code: str, seq: str, coords, w: float = 1):
	return (
		(
			pdb_code,
			[AA_TO_ID[aa] for aa in seq],
			coords,
			list(range(len(seq))),
			[],
			[]
		), None, None, w
	)
	
def collate_batch(p_batch):
	return Collate_Protein_Batch.collate(p_batch)


class EnzymeClassDataset(Dataset):

	def __init__(
		self,
		p_path = 'data',
		p_data_path = 'chains',
		p_dataset = 'training',
		p_fastafile = 'chain_list_pdb.fasta',
		p_random_seed = None,
		p_fold: str = None,        # particular fold from 1 to N
		p_train_mode = False, # to select all but the given fold (for training)
		p_data_aug = False,
		p_batch_pairs = False,
		p_load_data = False
	):
		if p_fold is not None and int(p_fold) < 1:
			raise Exception("Fold for CV should be a positive integer! Got: " + str(p_fold))
		
		# Random state.
		self.random_state_ = np.random.RandomState(p_random_seed)

		# Save the data augmentation parameters.
		self.data_augment_ = p_data_aug
		self.batch_pairs_  = p_batch_pairs
	
		# Get the paths.
		self.pdb_folder_ = os.path.join(os.path.join(p_path, p_data_path))
		pdb_fasta_file = os.path.join(p_path, p_fastafile)

		# Load the sequences from the fasta file
		self.list_chains_ = {}
		def process_fasta_file(fasta_file, folder):
			with open(fasta_file, 'r') as my_fasta_file:
				chain_name = ''
				for cur_line in my_fasta_file.readlines():
					if cur_line.startswith('>'):
						chain_name = cur_line.rstrip()[1:]
					else:
						cur_chain = cur_line.rstrip()
						cur_chain_ids = []
						for cur_aa in cur_chain:
							cur_chain_ids.append(AA_TO_ID[cur_aa])
						self.list_chains_[chain_name] = (np.array(cur_chain_ids), folder)

		process_fasta_file(pdb_fasta_file, self.pdb_folder_)
			
		# load datapoints
		self.datapoints_ = []
		with open(os.path.join(p_path, p_dataset+'.csv'), 'r') as labels_map_file:
			for cur_line in labels_map_file:
				line_split = cur_line.rstrip().split(',')
				line_split[2] = float(line_split[2])
				line_split[3] = float(line_split[3]) if line_split[3] else 1 # set default weight if not available
				# Cross-validation row selection
				if p_fold and (line_split[4] == p_fold) == p_train_mode:
					continue # do not include this fold
				self.datapoints_.append(line_split[:4]) # orig_pdb, mut_pdb, label, weight

		if p_load_data:
			self.data_ = []
			print()
			for cur_iter, cur_chain in enumerate(self.list_chains_):
				cur_path = os.path.join(cur_chain[2], cur_chain[0]+".npy")
				cur_pos_seq_path = os.path.join(cur_chain[2], cur_chain[0]+"_seq_pos.npy") 
				# cur_axes_path = os.path.join(cur_chain[2], cur_chain[0]+"_axes.npy")
				cur_aces_path = []
				self.data_.append((np.load(cur_path), np.load(cur_pos_seq_path), np.load(cur_axes_path)))
				if cur_iter%100==0:
					print("\r Loading {:6d}/{:6d}".format(cur_iter, len(self.list_chains_)), end ="")
			print()
		else:
			self.data_ = None


	def __len__(self):
		return len(self.datapoints_)

	def __getitem__(self, idx):
		orig_pdb, mut_pdb, label, weight = self.datapoints_[idx]
		
		orig_path = os.path.join(self.list_chains_[orig_pdb][1], orig_pdb +".npy")
		mut_path = os.path.join(self.list_chains_[mut_pdb][1], mut_pdb + ".npy")
		# cur_pos_seq_path = os.path.join(self.list_chains_[idx][2], self.list_chains_[idx][0]+"_seq_pos.npy")
		# cur_axes_path = os.path.join(self.list_chains_[idx][2], self.list_chains_[idx][0]+"_axes.npy")
		cur_axes_path = []
		
		noise = None
		
		def get_pdb(idx, cur_path, label: int):
			nonlocal noise
			
			cur_aa_ids = self.list_chains_[idx][0]
			if self.data_ is None:
				cur_pos = np.load(cur_path)
				# cur_seq_pos = np.load(cur_pos_seq_path)
				cur_seq_pos = list(range(len(cur_aa_ids)))
				cur_axes = []
			else:
				cur_pos = self.data_[idx][0]
				cur_seq_pos = self.data_[idx][1]
				cur_axes = self.data_[idx][2]

			cur_min = np.amin(cur_pos, axis=0, keepdims=True)
			cur_max = np.amax(cur_pos, axis=0, keepdims=True)
			center = (cur_max + cur_min)*0.5
			cur_pos = cur_pos - center
			
			if self.data_augment_:
				if noise is None or not self.batch_pairs_:
					noise = self.random_state_.normal(0.0, 0.05, cur_pos.shape)
				
				assert cur_pos.shape == noise.shape
				# print(cur_pos)
				cur_pos = cur_pos + noise
				# print(cur_pos)

			return idx, cur_aa_ids, cur_pos, cur_seq_pos, cur_axes
		
		return get_pdb(orig_pdb, orig_path, label), get_pdb(mut_pdb, mut_path, label), label, weight

		
		
		
