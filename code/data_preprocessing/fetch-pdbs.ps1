#!/usr/bin/env pwsh
# Downloads all the necessary PDBs from the source datasets
$ORIG_DIR = $pwd
cd $PSScriptRoot
. ./_include.ps1 # $RAW_DIR $PDB_DIR

$sources = (Import-Csv $RAW_DIR'soluprotmutdb_all_custom.csv'), (Import-Csv $RAW_DIR"fireprotdb_mutation_experiments_summary.csv" -Delimiter ';')

$pdbs_all = @()
foreach($data in $sources) {
	# getting all the unique PDBs in the dataset
	$pdbs = ($data | group {$_.pdb_id}).Name -join '|'
	$pdbs = $pdbs.Split('|') | Select -Unique
	
	$pdbs_all += $pdbs
}
$pdbs = $pdbs_all | where {$_ -ne ""} | select -unique # remove overlapping PDBs between datasets, if any

"Starting downloading {0} PDB files." -f $pdbs.Length
cd $PDB_DIR

# Downloading PDBs of wildtypes
foreach ($pdb in $pdbs) {
	Write-host -NoNewLine ("`r{0}/{1} " -f ++$i,$pdbs.Length) # progress counter
	
	if (Test-Path "$pdb.pdb") {
		Write-host -NoNewLine "$pdb.pdb already downloaded - skipped!"
	} else {
		# IDEA: but shouldn't be Biological assembly?
		Invoke-WebRequest -Uri https://files.rcsb.org/download/$pdb.pdb -OutFile "$pdb.pdb"
		Write-host -NoNewLine "Download of $pdb.pdb has finished!    "
	}
}

# return to the initial pwd
cd $ORIG_DIR
