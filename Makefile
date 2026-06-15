SPMv := v21_2

.PHONY: init prepare-mutants tests clean distribute webserver train

init: # initializing folder structure
	mkdir -p data_raw/pdbs
	mkdir -p data_temp/foldx data_temp/mutant-lists data_temp/mutated-pdbs data_temp/optimized-pdbs models/
	mkdir -p data_preprocessed

webserver:
	python app.py

# TRAINING
prepare-mutants:
	fetch-pdbs
	# mutants-lists
	# mutants
	rename-mutants
	reduce-dataset
	# process-pdbs

fetch-pdbs:
	pwsh code/data_preprocessing/fetch-pdbs.ps1
mutants-lists:
	pwsh code/data_preprocessing/create-mutants-list.ps1
mutants:
	pwsh code/data_preprocessing/create-mutant.ps1 # 4GKU
rename-mutants:
	pwsh code/data_preprocessing/rename-mutants.ps1 # 1AZV
reduce-dataset:
	pwsh code/data_preprocessing/reduce-dataset.ps1 # 1AZV
process-pdbs:
	python code/data_preprocessing/process.py

train:
	bash ./experiments/train.sh train-v21_2-on-TEM


# DEVELOPMENT
tests:
	python -m unittest discover tests/ -p "*.py" -v

watch-gpu:
	watch -n0.5 nvidia-smi

clean: # Removes SoluProtMut temp folder and __pycache__ folders
	rm -rf tmp/
	pwsh -Command "Get-ChildItem -Directory -Recurse -Include __pycache__ | Remove-Item -Force -Recurse"

# DISTRIBUTION
models/LGK.pth:
	python export-ensemble.py experiments/train-v21_2-on-LGK-dupl/model? --saveas $@
models/LGK-GFP.pth:
	python export-ensemble.py experiments/train-v21_2-on-LGK-GFP/model? --saveas $@
models/TEM\ (bypos).pth:
	python export-ensemble.py experiments/train-v21_2/bypos-on-TEM/model? --saveas "$@"

distribute: models/LGK.pth models/LGK-GFP.pth models/TEM\ (bypos).pth 
	rm -f SoluProtMut.zip
	tar -a -cvf SoluProtMut.zip --exclude="__pycache__" --exclude="SoluProtMut.zip" --exclude="Makefile" --exclude="tmp/" \
	app.py wrapper.py requirements.* README.md code/*.* code/data_preprocessing/*.py code/model_py/$(SPMv).py code/model_py/__init__.py code/model_py/utils.py models