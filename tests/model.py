import unittest

import numpy as np
import torch
from torch.testing import assert_close

from code.data_preprocessing.model import get_PDB
from code.data_preprocessing.process import process_pdb, mutate_seq
from code.data_loader import create_datapoint, collate_batch

from code.model_py import conf_dict, MutPred, PlaNNet, feed
from code.model_py.v21_2 import conf_dict, MutPred, PlaNNet, feed # test specific version of a model

from tests import *

# Test structure
PDB_CODE = "2ci2"; CHAIN = 'I'
LENGTH = 4 # length of protein to test

class TestInference(unittest.TestCase):
	
	@classmethod
	def setUpClass(self):
		# Structure preprocessing
		pdb_file = get_PDB(PDB_CODE)
		self.orig_seq, self.orig_coords = process_pdb(pdb_file, CHAIN)
		self.protein = create_datapoint(PDB_CODE, self.orig_seq, self.orig_coords)
		
		warnings.simplefilter('ignore', PDBConstructionWarning)
		
		pdb_code = '1EER'
		pdb_file = get_PDB(pdb_code)
		seq, coords = process_pdb(pdb_file, 'A')
		self.wildtype = create_datapoint(pdb_code, seq, coords)
		self.mutant   = create_datapoint(
			pdb_code,
			mutate_seq(seq, ['F', 'R'], [48, 150], ['D', 'D']),
			coords
		)
		
		self.model = PlaNNet()
		self.model.eval()
		
		conf_dict = torch.load('experiments/train-on-LGK-GFP/model0/best_model_acc.pth')
		self.predictor = MutPred(self.model)
		# self.predictor.load_state_dict(conf_dict['state_dict_prot_enc']) # without loading pre-trained model, the test is intialization-dependent (in theory)
		self.predictor.eval()
		

	def test_consistency(self):
		# tests the outputs are not batch-dependent (protein size, position 1st to 3rd)

		for i in list(range(1, 10)) + [-1, None]: # last two are full protein
			protein = create_datapoint(
				PDB_CODE,
				str(self.orig_seq)[:i],
				self.orig_coords[:i]
			)
		
			batch = collate_batch([protein, protein, protein])[:-2] # and drop weights and labels
			results, _ = feed(self.model, batch)
		
			with self.subTest(msg='Identity with protein of length %s' % i):
				assert_close(results[0], results[1])
				assert_close(results[0], results[2])
	
	def test_consistency_advanced(self):
		batch = collate_batch([
			self.wildtype, self.protein, self.mutant,
			self.protein, self.mutant, self.wildtype
		])[:-2]
		
		results, _ = feed(self.model, batch)
		assert_close(results[0], results[5])
		assert_close(results[1], results[3])
		assert_close(results[2], results[4])

	def test_antisymmetry(self):
		batch = collate_batch([self.wildtype, self.mutant])[:-2]
		results_dir, _ = feed(self.predictor, batch)

		batch = collate_batch([self.mutant, self.wildtype])[:-2]
		results_rev, _ = feed(self.predictor, batch)
		
		with self.subTest("Direct and reverse predictions: %g and %g" % (results_dir, results_rev)):
			# the reverse prediction should be inversed around 0,5 (i.e. forward mutation = 0.8, reverse mutation = 0.2)
			assert_close(results_dir, 1-results_rev)




if __name__ == '__main__':
	unittest.main()