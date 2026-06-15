from time import time

import torch

_current_milli_time = lambda: time() * 1000.0

# feeds batch to the predictor
def feed(model, cur_batch):
	start_time = _current_milli_time()
	chain_name, chain, chain_pos, chain_seq_pos, chain_axes, batch_ids = cur_batch

	dev = next(model.parameters()).device
	# move the batch to the same device as model is in
	chain         =         chain.to(dev, torch.int64)
	chain_pos     =     chain_pos.to(dev, torch.float32)
	chain_seq_pos = chain_seq_pos.to(dev, torch.int64)
	chain_axes    =    chain_axes.to(dev, torch.float32)
	batch_ids     =     batch_ids.to(dev)

	time_data = _current_milli_time() - start_time

	logits = model( # [batch_size] of residues each:
		chain,         # AA type (index)
		chain_pos,     # coordinates in 3D (of a-Carbon?)
		chain_seq_pos, # seqential index
		chain_axes,    # ??? axes (orthonormal) between consecutive residues
		batch_ids      # chain ID mask
	)
	
	return logits, time_data

class Feeder:
	# enables feeding models with batches in objective way: model.feed(batch) instead of imperative feed(model, batch)
	# simplifies imports...
	def feed(self, *args):
		return feed(self, *args)


class _Ensemble(torch.nn.Module, Feeder):

	def __init__(self,
		paths_or_n,
		base_nn: torch.nn.Module,
		mut_nn:  torch.nn.Module,
	):
		super().__init__()
		models = []

		if type(paths_or_n) is int:
			for i in range(paths_or_n):
				models.append(mut_nn(base_nn('avg', True, False))) # todo: this should not be hardcoded
		else:
			for path in paths_or_n:
				conf_dict = torch.load(path)
				model = mut_nn(
					base_nn(conf_dict['gl_pooling'], bool(conf_dict['embed_aa']), conf_dict['embed_aa'] == 'learn'))
				model.load_state_dict(conf_dict["state_dict_prot_enc"])
				model.eval()
				models.append(model)
		# all modules has to be properly registered as modules to be visible for PyTorch
		self.models = torch.nn.ModuleList(
			models)  # jit.trace would not work without this with a crpytic message: RuntimeError: "Cannot insert a Tensor that requires grad as a constant"

	def forward(self, *args):
		preds = [m(*args) for m in self.models]
		preds = torch.cat(preds)
		return torch.mean(preds)