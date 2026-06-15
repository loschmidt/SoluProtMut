# SoluProtMut 🔬
Prediction of a mutational effect on protein solubility.

You can access live demo on [Huggingface here](https://huggingface.co/spaces/loschmidt/SoluProtMut).

## Installation

Clone the [SoluProtMut repository](https://github.com/loschmidt/SoluProtMut).  
Then, extract the [FoldX](https://foldxsuite.crg.eu/products#foldx) distribution for your operating system into `external/foldx` in the cloned project folder, _make sure_ the binary is named `foldx`.  
Finally, install the dependencies and check the installation works:  
```bash
pip install -r requirements.in
python -m tests.inference
```

If the installation *fails*, or the application *crashes*, try to install *pinned dependencies* instead: `pip install -r requirements.txt`. Also check the table below.

## Getting the model weights
Model weights are not part of this repository.
We provide them upon request by an email address: soluprotmut (at) sci.muni.cz

After you obtain the weights, you put them in this directory and rename to `model_weights.pth`. 

#### Tested configurations:

| OS    | Python | PyTorch | Engine    | Project version                |
|-------|--------|---------|-----------|--------------------------------|
| Win 10 | 3.8.18 | 2.1.2   | CPU       | January 16, 2024               |
| Linux | 3.9.19 | 2.1.2   | CPU       | January 16, 2024               | 
| Linux | 3.8.13 | 1.10.0  | CUDA 11.3 | January 16, 2024                               | 

### Usage

```
wrapper.py [-h] [-v] pdb-code [chain] wild-type location mutation

Solubility change preditor:
  Predicts the change in the solubility of a given protein variant

positional arguments:
  pdb-code       PDB code
  chain          target chain          character; default=A
  wild-type      wild-type residue(s)  amino-acid[n]
  location       mutated position(s)   integer[n]
  mutation       1 or n substituents   amino-acid|amino-acid[n]

optional arguments:
  -h, --help     show this help message and exit
  -v, --verbose

Amino acids can be specified by both 1- or 3-letter code.
```

Example usage (double-point mutant in [erythropoietin](https://www.rcsb.org/sequence/1EER))`and its outpus:
```bash
./wrapper.py 1EER F,R 48,150 D --verbose
```

```
Predicted solubility change: 5.000398e-01 (solubilizing)
```

## Development

### Reproducible environment

`requirements.in` – project (direct) dependencies required for *inference*. Keep the PyTorch version specified here in sync with the link for the *pyg* wheels specified ibidem.

`requirements.txt` – _resolved_ and _pinned_ dependencies based on the file above, a result of `pip install -r requirements.in && pip freeze` in a fresh Python environment.  
note: `pip-compile` of pip-tools does not seem to work with complex PyTorch ecosystem 

### Style guide
#### Indentation
Use tabs `\t` for an indentation, spaces (after tabs) for an alignment ([reasoning](https://stackoverflow.com/a/5048130/1908192), [reasoning2](https://www.reddit.com/r/javascript/comments/c8drjo/nobody_talks_about_the_real_reason_to_use_tabs/
))

#### Naming conventions (PEP8-based)
module_name, CONSTANT_NAME, ClassName (capitalize first letters of words), subroutine_name, variable_name

ClassName, function_name: keep acronyms in their original case: GetPDB, get_PDB

 

