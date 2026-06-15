TMPDIR = "tmp/"

class PreprocessError(Exception):
	pass

from .process import process_pdb, mutate_seq
from .model import FoldX, get_PDB