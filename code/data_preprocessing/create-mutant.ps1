#!/usr/bin/env pwsh
#Requires -Version 7

<#
.SYNOPSIS
    Creates mutants specified in invidual_list files for proteins given in arguments
.EXAMPLE
    ./create-mutant.ps1 1biz
.NOTES
    Author: Jan Velecký velda@mail.muni.cz, 2023-24
#>
param(
	## Do not repair the structure with FoldX and use the raw structure
	[Parameter()] [switch] $NoRepair,
	[Parameter(ValueFromRemainingArguments=$true)] [string[]] $PDBList
)

# why is this necessary?
# Program 'foldx.exe' failed to run: StandardOutputEncoding is only supported when standard output is redirected.At line:15 char:2
# +     &$using:FOLDX_EXE --command=BuildModel --pdb-dir=$dir --pdb=$pdb ...
[console]::InputEncoding = [console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

cd $PSScriptRoot
. ./_include.ps1 # $MUTLIST_DIR $FOLDX_WORKDIR $FOLDX_DIR $FOLDX_EXE $PDB_DIR
$FoldX = "$FOLDX_DIR/foldx"

if(-not $PDBList) {
	$PDBList = @()
	
	ls $MUTLIST_DIR*.txt | foreach {
		$PDBList += $_.Name.Split('.')[1]
	}
}
$PDBList
break

$PDBList | foreach -throttlelimit 2 -parallel { # not supported in PS 5.1
	$p = $_
	$indir = $using:PDB_DIR
	$dir = "$using:FOLDX_WORKDIR" + $p.toUpper()
	mkdir -Force $dir # mkdir -p $dir
	$individuals = "${using:MUTLIST_DIR}individual_list.{0}.txt" -f $p
	$pdb = "{0}.pdb" -f $p
	
	if(-not $using:NoRepair) {
		if(-not (Test-Path ("{0}/{1}" -f $dir,$pdb) -NewerThan (Get-Item ("$indir/{0}.pdb" -f $p)).LastWriteTime)) {
			# # rm -Force $p/*
			&$using:FOLDX_EXE --command=RepairPDB --pdb-dir=$indir --pdb=$p.pdb --rotabaseLocation=$using:FOLDX_DIR/rotabase.txt --output-dir=$dir >$using:FOLDX_WORKDIR/$p.repair.log
		}
		# use the repaired structure in the next step
		$indir = $dir
		$pdb = "{0}_Repair.pdb" -f $p
	}
	# --noHeader=1 FoldX parameter does not have any effect
	&$using:FOLDX_EXE --command=BuildModel --pdb-dir=$indir --pdb=$pdb --rotabaseLocation=$using:FOLDX_DIR/rotabase.txt --output-dir=$dir --mutant-file=$individuals >$using:FOLDX_WORKDIR/$p.mutating.log
	# possibly should repair foldx generated files (remove headers)
}

# clean-up - remove empty folders (where FoldX failed to produce any files)
Get-ChildItem $FOLDX_WORKDIR -Directory |
	where { @( $_ | Get-ChildItem ).Count -eq 0 } |
	Remove-Item -Verbose