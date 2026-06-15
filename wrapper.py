#!/usr/bin/env python3
#---------------------------------------------------------------------
#---   Predictor of a protein solubility change given a mutation   ---
#---           by Jan Velecky velda@mail.muni.cz                   ---
#---              Loschmidt Laboratories, 2023-25                  ---
#---       example use:      python3 wrapper.py -h                 ---
#---------------------------------------------------------------------
import argparse
from functools import partial

import Bio.PDB.Polypeptide as AA

from code.data_preprocessing import get_PDB

# ---------------------------------------- DATA TYPES FOR ARGPARSE ----------------------------------------
def Type_index(value): # test 1UCY 1A
	try:
		if not value[-1].isdigit(): # insertion code
			ivalue = int(value[:-1])
		else:
			ivalue = int(value)
		return value
	except:
		raise argparse.ArgumentTypeError("%s is not a valid residue index" % value)

def Type_char(value):
	if len(value) != 1:
		raise argparse.ArgumentTypeError("'%s' is not a character" % value)
	return value

def Type_aminoAcid(value):
	orig = value
	value = str.upper(value)
	if len(value) == 1:
		try:
			value = AA.one_to_three(value)
		except:
			pass
	if not AA.is_aa(value):
		raise argparse.ArgumentTypeError("'%s' is not a valid amino-acid" % orig)
	return str.upper(AA.three_to_one(value))

def parseList(s, type):
	return [type(i) for i in s.split(',')]

def Type_listOf(basetype):
	return partial(parseList, type=basetype)

def Type_PDB(pdb_code):
	try:
		pdb_path = get_PDB(pdb_code.lower())
	except Exception as e:
		raise argparse.ArgumentTypeError(e)
	return pdb_code, pdb_path

def check_mutList(loc_list, orig_list, mut_list):
	if len(loc_list) != len(mut_list):
		if len(mut_list) == 1: # apply mutation to all positions
			mut_list *= len(loc_list)
	if len(loc_list) == len(mut_list) == len(orig_list):
		return mut_list
	else:
		raise ValueError("Inconsistent multi-point mutant specification")

# ----------------------------------------------END OF DATA TYPES ------------------------------------------

argParser = argparse.ArgumentParser(add_help = True,
	description=
"""Solubility change preditor:
  Predicts the change in the solubility of a given protein variant


""", epilog="""
Amino acids can be specified by both 1- or 3-letter code.

example of use: ./wrapper.py 1EER F,R 48,150 D --verbose
""",conflict_handler='resolve', # overwrite conflicts
	formatter_class=argparse.RawTextHelpFormatter,
	prefix_chars="+-",
	# exit_on_error=False # would be nice to use it for prompting a user for required params, but was implemented buggy: https://bugs.python.org/issue41255
)
argParser.add_argument(
	'input',
	metavar='pdb-code',
	type=Type_PDB,
	help="PDB code"
)
argParser.add_argument( # we only process one chain in the end
	'chain', nargs="?",
	default='A',
	type=Type_char,
	help="target chain          character; default=A"
)
argParser.add_argument(
	'orig',
	metavar='wild-type',
	type=Type_listOf(Type_aminoAcid),
	help="wild-type residue(s)  amino-acid[n]"
)
argParser.add_argument(
	'loc',
	metavar='location',
	type=Type_listOf(Type_index),
	help="mutated position(s)   integer[n]"
)
argParser.add_argument(
	'mut',
	metavar='mutation',
	type=Type_listOf(Type_aminoAcid),
	help="1 or n substituents   amino-acid|amino-acid[n]"
)
argParser.add_argument(
	'-v', '--verbose',
	action="store_true",
	#help="show ouputs from the underlaying tools"
)

argParser.add_argument('--ver', default=None, help=argparse.SUPPRESS)

# ---------------------------------------------- DATA PREDICTION ------------------------------------------

def check_input(args):
	pass

def predict(pdb_path, chain, orig, loc, mut, weights="model_weights.pth", version=None, rich_output=False):
	from code.predictor import EnsemblePredictor # expensive import left for after the argument check
	pred_model = EnsemblePredictor(weights=weights, version=version)

	try:
		assesment, prediction = pred_model.predict_change(pdb_path, chain, orig, loc, mut)
	except KeyError as e:
		raise ValueError("Non-existing position in the PDB: %s" % str(e))
		
	assesment = {'+': 'solubilizing ( > 0.5)', 'N': 'neutral (score ≈ 0.5)', '-': 'desolubilizing (score < 0.5)'}[assesment]
	
	# print(weights)
	
	if rich_output:
		# ternary gradient orange-black-blue
		ORANGE = (255, 165, 0)
		BLUESH = (100, 100, 255)
		color = tuple((o*max(0, 1-prediction*2) + b*max(0, (2*prediction)**2-1) for o, b in zip(ORANGE, BLUESH)))
		
		return f"Predicted solubiliztation score: <span style='color: rgb{color}'>{prediction:.2f} {assesment}</span>"
	else:
		return "Predicted solubiliztation score: %g (%s)" % (prediction, assesment)








if __name__ == '__main__':
	# ---------------------------------------- ARGUMENTS PROCESSING ----------------------------------------
	args = argParser.parse_args()

	# positional arguments parsing (argparse can't cope with nested positional arguments)
	try:
		args.mut = check_mutList(args.loc, args.orig, args.mut)
	except Exception as e:
		argParser.error(str(e))
	pdb_code, pdb_path = args.input
	chain = args.chain

	# import code.data_preprocessing.model as modeling
	# if(args.verbose):
		# modeling.VERBOSE_LEVEL = modeling.VERBOSE_VERBOSE

	# ------------------------------------- PREPROCESSING & INFERENCE --------------------------------------
	print()
	print(predict(pdb_path, chain, args.orig, args.loc, args.mut))