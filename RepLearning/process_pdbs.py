# author: Pedro
import os
import numpy as np
import gzip
import pickle

# should not contain multi-chains
# should deal with partial residues
# mutated residues should be in the structure

import warnings
warnings.filterwarnings("ignore")

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import CaPPBuilder
from Bio.PDB.Polypeptide import is_aa, three_to_one

parser = PDBParser(PERMISSIVE=1)

list_pdbs = []
# for pdb_path in ['./data_pdb/']:
for pdb_path in ['./FoldX-optimized/', './FoldX-structures/']:
    list_pdbs += [os.path.join(pdb_path, f) for f in os.listdir(pdb_path) if os.path.isfile(os.path.join(pdb_path, f))]

total_chains = 0

os.makedirs("./data_chains/pdb_chains/", exist_ok=True) # folders for preprocessed data
with open("./data_chains/chain_list_pdb.fasta", 'w') as chain_list_file:

    for cur_iter, cur_pdb in enumerate(list_pdbs):

        structure_id = cur_pdb.split('/')[-1][:-4]

        if cur_iter% 100 == 0:
            print(cur_iter, len(list_pdbs), structure_id)

        with open(cur_pdb, 'rt') as ifh:
            structure = parser.get_structure(structure_id, ifh)
    
            for chain in structure[0]:
                print(structure_id)
                chain_id = chain.id
                res_types = []
                res_pos = [] 
                for residue in chain:
                    if is_aa(residue.get_resname()):
                        if 'CA' in residue:
                            try:
                                res_types.append(three_to_one(residue.get_resname()))
                            except KeyError as e:
                                res_types.append('X')
                            atom_ca = residue['CA']
                            res_pos.append(atom_ca.get_coord())

                res_pos = np.array(res_pos)
                seq = "".join(res_types)

                if len(res_types) > 10:

                    total_chains += 1

                    chain_list_file.write(">"+structure_id+"."+chain_id+"\n")
                    chain_list_file.write(seq+"\n")
                
                    np.save("./data_chains/pdb_chains/"+structure_id+"."+chain_id, res_pos)

print("Total chains:", total_chains)
