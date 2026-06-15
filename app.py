#---------------------------------------------------------------------
#---              Launches SoluProtMut web UI                      ---
#---           by Jan Velecky velda@mail.muni.cz                   ---
#---              Loschmidt Laboratories, 2025                     ---
#---                startup: python app.py                         ---
#---------------------------------------------------------------------
import gradio as gr
from Bio.PDB.PDBParser import PDBParser

from wrapper import *

parser = PDBParser(PERMISSIVE=1)

def predict_solubility(pdb_code, chain, orig, loc, mut, model, version=None):
	global MODELS
	try:
		pdb_code, pdb_path = Type_PDB(pdb_code)
		loc_list = parseList(loc, Type_index)
		orig_list = parseList(orig, Type_aminoAcid)
		mut_list = parseList(mut, Type_aminoAcid)

		mut_list = check_mutList(loc_list, orig_list, mut_list)

		return predict(pdb_path, chain, orig_list, loc_list, mut_list, weights=MODELS[model], rich_output=True)
	except Exception as e:
		return f"Error: {str(e)}"

_pdb_code = None
_chains = None
def get_chains(pdb_code): # display chains stored in the PDB structure
	global _pdb_code, _chains
	
	if pdb_code != _pdb_code:
		_pdb_code = pdb_code
		try:
			pdb_code, pdb_path = Type_PDB(pdb_code)
			structure = parser.get_structure(pdb_code, pdb_path)
			_chains = [ch.id for ch in structure[0]]
			return gr.update(choices=_chains, value=_chains[0])
		except Exception as e:
			raise gr.Error(str(e))
	return gr.update(choices=_chains)

from pathlib import Path
MODELS = sorted(list((Path(__file__).parent / "models").iterdir()), key=lambda m: m.stem)




# --- Gradio Interface ---
with gr.Blocks(
	title="SoluProtMut", css="""
		.gradio-container {
			max-width: 900px !important
		} blockquote {
			margin: 1em !important
		}
	""") as demo:
	with gr.Row():
		gr.HTML()
		gr.HTML(value="<img src='https://loschmidt.chemi.muni.cz/portal/soluprotmut_logo.png'>")
	gr.Markdown("""
		## SoluProtMut: prediction of a mutational effect on protein solubility
		specify the mutation in the protein of interest:""")

	with gr.Row():
		pdb_code = gr.Textbox(label="PDB Code",
			placeholder="1EER",
			max_length=4 # 12 # new PDB identifier has a shape of: pdb_00001abc https://proteopedia.org/w/PDB_code
		)
		chain = gr.Radio(choices=[], label="Chain", scale=1)

	with gr.Row():
		loc = gr.Textbox(label="Mutated position(s)", placeholder="48,150")
		orig = gr.Textbox(label="Wild-type residue(s)", placeholder="F,R", scale=0)
		mut = gr.Textbox(label="Mutant residue(s)", placeholder="D[,A]", scale=0)

	# with gr.Accordion("Model selection"):
	model_names = [m.stem for m in MODELS]
	assert model_names[0] == "LGK"
	model_names[0] += " (recommended)"
	model = gr.Radio(choices=model_names, label="Model selection", type="index", value=model_names[0])
	
	output = gr.HTML()

	_pdb_code = None
	pdb_code.blur(fn=get_chains, inputs=[pdb_code], outputs=chain)
	pdb_code.submit(fn=get_chains, inputs=[pdb_code], outputs=chain)

	predict_btn = gr.Button("Predict solubility effect", variant='primary', size='lg', scale=0)
	# predict_btn.style(full_width=False)
	dict_submit = {
		'fn': predict_solubility,
		'inputs': [pdb_code, chain, orig, loc, mut, model],
		'outputs': [output]
	}
	# submit by entering in the text boxes or by the submit button
	loc.submit(**dict_submit)
	orig.submit(**dict_submit)
	mut.submit(**dict_submit)
	predict_btn.click(**dict_submit)
	
	examples = gr.Examples(
		examples=[
			["1EER", "48,150", "F,R", "D"],
			["1EER", "13", "E", "K"],
			["1z0q", "19", "F", "A"],
			["1col", "86", "W", "F"],
		],
		# example_labels = ["1EER F48D,R150D"],
		inputs=[pdb_code, loc, orig, mut],
		label="Examples (click on a line to pre-fill the inputs)",
		cache_examples=False
	)
	examples.load_input_event.then(fn=get_chains, inputs=pdb_code, outputs=chain)
	
	gr.Markdown(value="""
	<br/>
	
	**Acknowledgement**. Please, use the following citation to acknowledge the use of our tool:  
	> Velecký, J., Faldynová H., Hermosilla, P., Sandlerová, N., Doerr, M., Egersdorfová, S., Bornscheuer, U., Prokop, Z., Damborský, J., Mazurenko, S., 2025:
	> SoluProtMut: Siamese Deep Learning for Solubility Effect Prediction in Protein Mutations and Experimental Validation.
	> *In preparation.*
	""")

	demo.launch(inbrowser=True)
