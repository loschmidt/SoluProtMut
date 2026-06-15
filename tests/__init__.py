import unittest, warnings, functools

from Bio.PDB.PDBExceptions import PDBConstructionWarning # discontinuation of chains
from torch.testing import assert_close

assert_close = functools.partial(assert_close, atol=1e-6, rtol=1e-6)