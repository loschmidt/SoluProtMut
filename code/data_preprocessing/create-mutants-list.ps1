#!/usr/bin/env pwsh
# Prepares mutation files for FoldX and filenames mapping

cd $PSScriptRoot
. ./_include.ps1 # Get-PDBs $MUTLIST_DIR $MAP_FILE

# $data = Import-Csv .\fireprotdb_all.csv # contains duplicates
# $data = Import-Csv ${RAW_DIR}soluprotmutdb_all_custom.csv
$data = Import-Csv ${RAW_DIR}fireprotdb_mutation_experiments_summary.csv -Delimiter ';' # custom export from stage which is correct

# getting all the unique PDBs in the dataset
$pdbs = Get-PDBs $data
"Total # of unique PDBs: {0}" -f $pdbs.Length

$n_all = $data.length
# filter out the datapoints with no PDB structure (chain)
$data_filtered = $data | where {$_.pdb_id -ne "" -and $_.chain -ne ""}
$n_w_struct = $data_filtered.length

# filter out the datapoints without mutation positions mapped to their PDBs
$data_filtered = $data_filtered | where {
	$_.pdb_position -and
	($_.pdb_position -split ',').length -eq ($_.mutation -split ',').length
}
$n_pdbs_filtered = (Get-PDBs $data_filtered).Length

# count without duplicate variants
$n_wstr_uni = ($data_filtered | Sort-Object pdb,mutations -unique).Length

"Total # of datapoins: {0}, with structure: {1} ({4}), of which map: {2} ({5}), which unique: {3}" -f
	$n_all, $n_w_struct, $data_filtered.length, $n_wstr_uni, $pdbs.Length, $n_pdbs_filtered

$data_filtered = $data_filtered | Sort-Object pdb,{ [regex]::replace($_.pdb_position, '\d+', { $args[0].Value.PadLeft(4) })},mutation # natural sort


# MUTATION CODES CREATION
$muts_counter = @{}
Remove-Item $MUTLIST_DIR/* # todo: confirm before deleting?
"ac,pdb,foldx_file,mutations" >$MAP_FILE
# todo: make the generated map file sorted
"Looping through mutations codes..."
&{ # assemble all rows and put it into the file online (without reopenings -> faster)
	foreach ($p in $data_filtered) {
		$pdb = $p.pdb_id.split('|')[0] # only take a first PDB variant
		
		$positions    = $p.pdb_position -split ','
		$wildtypes    = $p.wild_type -split ','
		$substituents = $p.mutation -split ','
		$FoldX_muts   = $null
		$muts_codes   = $null
		foreach($i in 0..($positions.length-1)) {
			$mut_code = "{0}{1}{2}" -f $p.chain, $positions[$i], $substituents[$i]
			$muts_codes = ,$mut_code
			$FoldX_muts += ,("{0}$mut_code" -f $wildtypes[$i])
		}
		
		$FoldX_codes = $FoldX_muts -join ','
		try {
			$index = $muts_counter[$pdb].IndexOf($FoldX_codes)
		} catch {
			$index = -1
		}
		if($index -eq -1) {
			$index = $muts_counter[$pdb].Length
			$muts_counter[$pdb] += ,$FoldX_codes
		}
		
		@(
			$(if ($p.result_id) {$p.result_id} else {$p.experiment_id}),
			$pdb,
			"${pdb}_Repair_$($index+1).pdb",
			# "${pdb}_$($index+1).pdb",
			($FoldX_muts -join '+') # not necessary to store mutated chain
			# ($muts_codes -join '+')
		) -join ','
	}
} >>$MAP_FILE # write down filename mappings

"Saving mutation files..."
foreach ($pdb in $muts_counter.keys) {
	($muts_counter[$pdb] -join ";`n") + ';' >("$MUTLIST_DIR/individual_list.{0}.txt" -f $pdb)
}

