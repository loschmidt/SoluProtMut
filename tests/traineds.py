import unittest

from code.data_preprocessing import get_PDB
from code.predictor import EnsemblePredictor

from tests import *

TEST_POINTS = [
	('1EER', 'A', (['F'], [48], ['D'])),
	('1EER', 'A', (['F', 'R'], [48, 150], ['F', 'R'])), # self-mutant: neutral
	('1EER', 'A', (['F', 'R'], [48, 150], ['D', 'D'])),
	('1ANF', 'A', (['G', 'I'], [32, 33], ['D', 'P'])),
	('1LNI', 'A', (['T'], [76], ['A'])),
]


class TestTrainedModels(unittest.TestCase):
	
	def setUp(self):
		warnings.simplefilter('ignore', PDBConstructionWarning)
	
	def check(self, predictor, expected):
		for (PDB, chain, mut), exp in zip(TEST_POINTS, expected):
			with self.subTest("Mutation: %s, expected: %s" % (PDB + ' ' + chain + ' ' + str(mut), exp)):
				pdb_path = get_PDB(PDB)
				pred = predictor.predict_change(pdb_path, chain, *mut)
				
				self.assertEqual(pred[0], exp[0]) # label
				assert_close(pred[1], exp[1])     # raw prediction score
	
	def test_v21_2_LGK_GFP(self):
		predictor = EnsemblePredictor(version='21_2')
		
		EXPECTED = [('+', 0.5132869482040405), ('N', 0.5), ('-', 0.1167784184217453),
			('+', 0.510742),
			('-', 0.379567),
		]
		self.check(predictor, EXPECTED)
	
	def test_v21_1_LGK_GFP(self):
		predictor = EnsemblePredictor(version='21_1')
		
		EXPECTED = [('N', 0.5035609006881714), ('-', 0.4910307824611664), ('-', 0.1248282790184021),
			('N', 0.49888959527015686),
			('-', 0.07750764489173889),
		]
		self.check(predictor, EXPECTED)


if __name__ == '__main__':
	unittest.main()