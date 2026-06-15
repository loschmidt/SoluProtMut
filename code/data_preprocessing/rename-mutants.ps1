#!/usr/bin/env pwsh
# Renames FoldX optimized and mutatated files to meaningful names: <PDB-ID>.<MUTLIST>.pdb
# ./create-mutant.ps1 [PDB1 [PDB2 [...]]
# ex. use: ./rename-mutants.ps1 1biz

cd $PSScriptRoot
. ./_include.ps1 # $MAP_FILE $MUTANTS_DIR $OPTIMS_DIR $FOLDX_WORKDIR

$variants = Import-CSV $MAP_FILE
if($Args) {
	$variants = $variants | where pdb -in $Args
}

# pick up and rename optimized structures
foreach($pdb in $variants | Select pdb -unique) {
	$new_filename = "$($pdb.pdb.ToLower()).pdb"
	$raw_filepath = $FOLDX_WORKDIR + $pdb.pdb + '/' + $pdb.pdb + '_Repair.pdb'

	# copy -Force may be needed
	copy $raw_filepath $OPTIMS_DIR$new_filename -ErrorAction Ignore
}

"Trying to find & rename {0} FoldX-generated mutant structures..." -f $variants.Length

# pick up and rename mutated structures
foreach($datapoint in $variants) {
	$new_filename = "$($datapoint.pdb.ToLower()).$($datapoint.mutations).pdb"
	$raw_filepath = $FOLDX_WORKDIR + $datapoint.pdb + '/' + $datapoint.foldx_file
	
	copy $raw_filepath $MUTANTS_DIR$new_filename -ErrorAction Ignore
}
