from .v21_2 import PlaNNet, MutPred, Ensemble, conf_dict, feed

# Print number of parameters.
def get_n_params(model):
	pp=0
	for p in model.parameters():
		nn=1
		for s in list(p.size()):
			nn = nn*s
		pp += nn
	return pp

def print_n_params(model, out_f = print):
	out_f("Number of parameters: ", get_n_params(model))
