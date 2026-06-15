#!/usr/bin/env pwsh
# Reduces the dataset with the datapoints having truly available their mutated structures
# ./reduce-dataset.ps1 [PDB1 [PDB2 [...]]
# ex. use: ./reduce-dataset.ps1 1azv

. $PSScriptRoot/_include.ps1 # $RelRoot $MAP_FILE

$dataset = "soluprotmutdb_all_custom.csv"
$data = Import-Csv $RAW_DIR$dataset
# $dataset = "fireprotdb_mutation_experiments_summary.csv"
# $data = Import-Csv $RAW_DIR$dataset -Delimiter ';'

$reduced_output = "{0}/$dataset" -f (Resolve-Path -Relative $TEMP_DIR)

# filter datapoints for specified PDBs (if specified)
if($Args) {
	$pdbs = $Args # $Args will get shadowed in the cmdlet below
	$data = $data | Where-Object { $_.pdb_id -and $_.pdb_id.substring(0,4) -in $pdbs }
}

$data = $data |
	Foreach-Object {
		$mutcode = Create-Mutcode $_ -FoldX $True
		$mutcode2 = $mutcode -replace "^.", "" # without 1st letter (wt AA)
		foreach($mutcode in ($mutcode, $mutcode2)) { # check both possible conventions
			$file = "{0}.{1}.pdb" -f $_.pdb_id.ToLower(), $mutcode
			
			if(Test-Path "$MUTANTS_DIR$file") {
				$_ | Add-Member -MemberType NoteProperty -Name 'pdb_file' -Value $file -PassThru
				break
			}
		}
	} |	Where {
		$_.pdb_file
	}

"Datapoints with a mutated structure available ({1}) exported into `"{0}`"" -f $reduced_output,$data.Length
$data | Export-Csv -UseQuotes AsNeeded $reduced_output