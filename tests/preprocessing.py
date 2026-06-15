import unittest

from numpy.testing import assert_allclose 

from code.data_preprocessing.model import get_PDB
from code.data_preprocessing.process import process_pdb, mutate_seq, Type_Seq

from code.data_loader import create_datapoint, collate_batch

from tests import *

# Test structure
PDB_CODE = "1eer"; CHAIN = 'A'

class TestPreprocessing(unittest.TestCase):
	
	@classmethod
	def setUpClass(self):
		# Structure preparation
		self.pdb_file = get_PDB(PDB_CODE)
		warnings.simplefilter('ignore', PDBConstructionWarning)
		
	def test_(self): # test on continuous (space-less) multi-chain protein, indexing from 1
		seq, coords = process_pdb(self.pdb_file, CHAIN)
		
		# order of the following two tests is important:
		#   mutating first to see that the input sequence is immutable
		with self.subTest("Mutating sequence"):
			mut_seq: Type_Seq = mutate_seq(seq, ['F', 'R'], [48, 150], ['D', 'D'])
			
			self.assertEqual(str(mut_seq), 'APPRLICDSRVLERYLLEAKEAEKITTGCAEHCSLNEKITVPDTKVNDYAWKRMEVGQQAVEVWQGLALLSEAVLRGQALLVKSSQPWEPLQLHVDKAVSGLRSLTTLLRALGAQKEAISNSDAASAAPLRTITADTFRKLFRVYSNFLDGKLKLYTGEACRTGDR')
		
		with self.subTest("Loading of seqeunce and coordinates (1EER)"):
			self.assertEqual(str(seq), 'APPRLICDSRVLERYLLEAKEAEKITTGCAEHCSLNEKITVPDTKVNFYAWKRMEVGQQAVEVWQGLALLSEAVLRGQALLVKSSQPWEPLQLHVDKAVSGLRSLTTLLRALGAQKEAISNSDAASAAPLRTITADTFRKLFRVYSNFLRGKLKLYTGEACRTGDR')
			self.assertEqual(seq[48], 'F')
			assert_allclose(coords[10-1], [-20.475, 15.498, 99.484]) # 10th C-a in the PDB
			assert_allclose(coords[150-1], [-19.577, 26.599, 105.913])



if __name__ == '__main__':
	unittest.main()