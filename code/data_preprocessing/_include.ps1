# Directories
$RelRoot = $PSScriptRoot + '/../..' | Resolve-Path -Relative # Project root directory

$MUTLIST_DIR   = $RelRoot + '/data_temp/mutant-lists/'

$RAW_DIR       = $RelRoot + '/data_raw/'
$PDB_DIR       = $RelRoot + '/data_raw/pdbs/'

$TEMP_DIR      = $RelRoot + '/data_temp/'
$FOLDX_WORKDIR = $RelRoot + '/data_temp/foldx/'
$OPTIMS_DIR    = $RelRoot + '/data_temp/optimized-pdbs/'
$MUTANTS_DIR   = $RelRoot + '/data_temp/mutated-pdbs/'

# Files
$MAP_FILE = "$MUTLIST_DIR/!spmdb-foldx-map.csv" # for mapping between the datasets and the files from FoldX

# Executables
$FOLDX_DIR = $RelRoot + '/external/foldx/'
$FOLDX_EXE = "${FOLDX_DIR}foldx"
# Since 2024 32-bit version of FoldX no longer available
# if([System.Environment]::OSVersion.Platform -eq "Win32NT") {
	# $FOLDX_EXE = "${FOLDX_DIR}foldx-32" # 64-bit version is buggy a lot on Windows
# }

# Constants
$PSDefaultParameterValues['Out-File:Encoding'] = 'ascii'
$AA_CODES = @{ # conversions of amino-acids 1-letter and 3-letter codes
	CYS = 'C'; 'C' = 'CYS'
	ASP = 'D'; 'D' = 'ASP'
	SER = 'S'; 'S' = 'SER'
	GLN = 'Q'; 'Q' = 'GLN'
	LYS = 'K'; 'K' = 'LYS'
	ILE = 'I'; 'I' = 'ILE'
	PRO = 'P'; 'P' = 'PRO'
	THR = 'T'; 'T' = 'THR'
	PHE = 'F'; 'F' = 'PHE'
	ASN = 'N'; 'N' = 'ASN'
	GLY = 'G'; 'G' = 'GLY'
	HIS = 'H'; 'H' = 'HIS'
	LEU = 'L'; 'L' = 'LEU'
	ARG = 'R'; 'R' = 'ARG'
	TRP = 'W'; 'W' = 'TRP'
	ALA = 'A'; 'A' = 'ALA'
	VAL = 'V'; 'V' = 'VAL'
	GLU = 'E'; 'E' = 'GLU'
	TYR = 'Y'; 'Y' = 'TYR'
	MET = 'M'; 'M' = 'MET'
}

# Functions
function Get-PDBs {
	# will return list of unique PDBs from a dataset
	# a datapoint may have specified multiple PDBs delimited by |
	param (
		$dataframe,
		$delimiter = '|'
	)
	
	$pdbs = ($dataframe | group {$_.pdb_id}).Name -join $delimiter
	$pdbs.Split('|') |
		select -Unique |
		where {$_ -ne ""}
}

function Create-Mutcode {
	# will create a mutation code for a given datapoint
	param (
		$dp,            # dataframe DataPoint
		$delimiter = '+',
		$FoldX = $True  # whether FoldX format: <wt><chain><pos><mut> (otherwise: <wt><pos><mut>)
	)
	
	$positions    = $( If($dp.pdb_position -ne "") {$dp.pdb_position} else {$dp.position} ) -split ','
	$wildtypes    = $dp.wild_type -split ','
	$substituents = $dp.mutation -split ','
	$chain        = $dp.chain
	$mut_codes    = @()
	$FoldX_muts   = @()
	
	foreach($i in 0..($positions.length-1)) {
		$wt  = $wildtypes[$i]
		$pos = $positions[$i]
		$mut = $substituents[$i]
		
		$mut_codes = ,("{0}{1}{2}" -f $wt, $pos, $mut)
		
		$FoldX_muts += ,("{0}{1}{2}{3}" -f $wt, $chain, $pos, $mut)
	}
	
	if($FoldX) {
		return $FoldX_muts -join $delimiter
	}
	return $mut_codes -join $delimiter
}

