import os.path, sys

from . import TMPDIR, PreprocessError
os.makedirs(TMPDIR, exist_ok=True)

FILEDIR = os.path.dirname(__file__)
FOLDX_DIR      = FILEDIR + "/../../external/foldx/"
FOLDX_BINARY   = FOLDX_DIR + "foldx"
# if sys.platform == "win32":
	# FOLDX_BINARY   = FOLDX_DIR + "foldx-32"
FOLDX_ROTAFILE = os.path.relpath(FOLDX_DIR) + "/rotabase.txt"
# os.environ["FOLDX_BINARY"] = os.path.dirname(__file__) + "/external/foldx/foldx-32.exe"

# verbosity of logs
VERBOSE_IMPORTANT = 0
VERBOSE_VERBOSE = 1
VERBOSE_LEVEL = VERBOSE_IMPORTANT

def echo(txt="", level=VERBOSE_VERBOSE):
	if(level <= VERBOSE_LEVEL):
		print(txt)

def echoStage(name, exe = None):
	echo()
	name = " %s " % name
	TEXT = "predictor:----------------------------------------------------------------------"
	hyphen_len = int((len(TEXT) - len(name)) / 2)
	TEXT = TEXT[0:hyphen_len] + name + TEXT[-hyphen_len:]
	echo(TEXT)
	if exe:
		echo("Running: %s" % exe)
	else:
		echo()

def get_PDB(pdb_code: str, force: bool = False):
	# todo: make filename small
	from urllib import request, error
	global TMPDIR

	pdb_filename = pdb_code + ".pdb"
	pdb_filepath = TMPDIR + pdb_filename
	if force or not os.path.isfile(pdb_filepath):
		try:
			request.urlretrieve('https://files.rcsb.org/download/' + pdb_filename, pdb_filepath)
		except error.HTTPError as e:
			if e.code == 404:
				raise PreprocessError("PDB file not available online – '%s': %s" % (pdb_code, e.url))
			raise

	return pdb_filepath

    
def exec(cmd, level=VERBOSE_VERBOSE, **args):
	import subprocess
	print(cmd)
	
	pipe = None if level <= VERBOSE_LEVEL else subprocess.DEVNULL
	return subprocess.call(cmd, shell=True, stdout=pipe, **args)

def FoldX(file,
	chain: str = None,
	wildtype: list = None,
	location: list = None, 
	mutation: list = None,
	force: bool=False
):
	global TMPDIR, FOLDX_BINARY, FOLDX_ROTAFILE
	
	if(exec("\"%s\" -h" % FOLDX_BINARY)):
		raise PreprocessError("FoldX is not installed or has expired!")
	
	dname = os.path.dirname(file) or '.'
	fname = os.path.basename(file)

	# prepare PDB structure for run in FoldX
	rname = "_Repair".join(os.path.splitext(fname)) # repaired PDB filename
	rpath = TMPDIR + rname
	if force or not os.path.isfile(rpath):
		print(TMPDIR)
		exe = "\"%s\" --command=RepairPDB --pdb-dir=\"%s\" --pdb=\"%s\" --rotabaseLocation=\"%s\" --output-dir=\"%s\"" % \
			(FOLDX_BINARY, dname, fname, FOLDX_ROTAFILE, TMPDIR)
		echoStage("EXECUTING FOLDX AS MUTATION ENGINE", exe)
		# Repair PDB
		exit_code = exec(exe)
		if exit_code:
			raise PreprocessError("FoldX optimization failed with code: %i" % (exit_code))
			return None, None

	if not mutation:
		return rpath, None

	# prepare mutation list for FoldX
	ifile = TMPDIR + 'individual_list.txt'
	muts = []
	for w,l,m in zip(wildtype, location, mutation):
		muts.append("%s%s%s%s" % (w,chain,l,m))
	with open(ifile, 'w') as individuals:
		individuals.write(','.join(muts) + ';')

	# Create mutants
	# --numberOfRuns=5 does not seems to change structure a lot, does not influence the backbone
	exe = "\"%s\" --command=BuildModel --pdb-dir=\"%s\" --pdb=\"%s\" --rotabaseLocation=\"%s\" --output-dir=\"%s\" --mutant-file=\"%s\"" % \
		(FOLDX_BINARY, TMPDIR, rname, FOLDX_ROTAFILE, TMPDIR, ifile)
	exit_code = exec(exe)
	# exit_code = 0
	if exit_code:
		raise PreprocessError("FoldX mutation failed with code: %i" % (exit_code))
		return rpath, None

	mname = TMPDIR + "_1".join(os.path.splitext(rname)) # 1st mutant of inidivudals file has "_Repair_1" suffix
	
	# name of optimized original file, name of mutated file
	return rpath, mname
