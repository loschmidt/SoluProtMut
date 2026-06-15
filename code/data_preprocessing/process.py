# author: Jan Velecky, adapted from Pedro
# creates coordinate (.npy) & sequence (.fasta) files from the folders containing (mutated or not) PDB files
import os
import numpy as np
import gzip
import pickle

ROOT_DIR    = os.path.dirname(__file__) + '/../../'
RAW_DIR  = ROOT_DIR + '/data_raw/pdbs/'
OPTIMS_DIR  = ROOT_DIR + '/data_temp/optimized-pdbs/'
MUTANTS_DIR = ROOT_DIR + '/data_temp/mutated-pdbs/'

PREPRO_DIR  = ROOT_DIR + '/data_preprocessed/'
CHAINS_DIR  = PREPRO_DIR + '/chains/'

# should not contain multi-chains
# should deal with partial residues
# mutated residues should be in the structure

import warnings
warnings.filterwarnings("ignore")

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa, three_to_one

parser = PDBParser(PERMISSIVE=1)

# a dict of {pos: aa} with some str compability
class Type_Seq(dict):
	def __init__(self, *arg, **kw):
		super().__init__(*arg, **kw)
	
	def __str__(self): # directly convertible to amino-acid sequence string
		return ''.join(self.values())
	
	def __repr__(self):
		return str(dict(self))
	
	def __iter__(self): # dict default is over the keys
		yield from self.values()
	
	def __add__(self, other):
		if type(other) is str:
			return str(self) + other
	
	# def __getitem__(self, subscript): # slicing would be nice to have, but should it slice by indexes or as sequence? Who knows.
		# if isinstance(subscript, slice):
			# return self.items() print(subscript.start, subscript.stop, subscript.step)
		# else:
			# return dict[subscript]
	
	def copy(self): # sing super's (dict's) copy would create another dict, not a Type_Seq
		return Type_Seq(dict(self))


def _process_pdb(filepath, gap_detect=True):
	structure_name = filepath.split('/')[-1][:-4]

	with open(filepath, 'rt') as ifh:
		structure = parser.get_structure(structure_name, ifh)

		for chain in structure[0]:
			res_aas = {} # sequence from the pdb
			res_pos = [] # positions of Cα
			seqid1 = None
			for residue in chain:
				if is_aa(residue.get_resname()):
					if 'CA' in residue:
						_, seqid, ins_code = residue.id
						if(gap_detect):
							if seqid1:
								gap = seqid - seqid1
								if gap > 8:
									print("Gap of %i AAs detected at %s" % (gap, str(residue.full_id)))
							seqid1 = seqid

						seqid = (str(seqid) + ins_code).strip() # remove ' ' for an empty insertion code
						try: # std amino acids
							res_aas[seqid] = three_to_one(residue.get_resname())
						except KeyError as e:
							res_aas[seqid] = 'X'
						atom_ca = residue['CA']
						res_pos.append(atom_ca.get_coord())

			res_pos = np.array(res_pos)
			
			yield chain.id, Type_Seq(res_aas), res_pos


def process_pdb(filepath, chain: str = None):
	chains = _process_pdb(filepath)
	if not chain: # return generator (for all chains)
		return chains
		
	for ch in chains: # return the specified chain
		if ch[0] == chain:
			return ch[1:]


def mutate_seq(res_aas: Type_Seq, wildtype: list, location: list, mutation: list) -> Type_Seq:
	res_aas = res_aas.copy()
	
	for w,l,m in zip(wildtype, location, mutation):
		if res_aas[l] != w:
			raise ValueError("Wildtype residue mismatch: %s is actually %s" % (''.join((w,str(l))), res_aas[l]))
		res_aas[l] = m
	
	return res_aas


if __name__ == "__main__":

	list_pdbs = []
	set_pdbs = set()
	# process all PDBs
	for pdb_path in [OPTIMS_DIR, RAW_DIR, MUTANTS_DIR]:
		# use RAW structure if optimized not found
		pdbs = {f for f in os.listdir(pdb_path) if os.path.isfile(os.path.join(pdb_path, f))}
		pdbs -= set_pdbs
		set_pdbs |= pdbs # merge sets
		list_pdbs += [os.path.join(pdb_path, f) for f in pdbs]

	total_chains = 0
	os.makedirs(CHAINS_DIR, exist_ok=True) # folders for preprocessed data
	
	with open(PREPRO_DIR+"chain_list_pdb.fasta", 'w') as chain_list_file:
		for cur_iter, cur_pdb in enumerate(list_pdbs):
			structure_name = cur_pdb.split('/')[-1][:-4].lower()
			
			if cur_iter% 100 == 0:
				print("%i/%i %s" % (cur_iter, len(list_pdbs), structure_name))
			
			for chain_id, seq, res_pos in process_pdb(cur_pdb): # todo: _process_pdb(cur_pdb, gap_detect=False)
				if len(seq) > 10:
					total_chains += 1

					chain_list_file.write(">"+structure_name+"."+chain_id+"\n")
					chain_list_file.write(seq+"\n")
				
					np.save(CHAINS_DIR+structure_name+"."+chain_id, res_pos)

	print("Total chains:", total_chains)