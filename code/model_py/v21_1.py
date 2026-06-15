import signal

# RUN CONFIGURATION
VERSION = "21-mutpred"
VERSION_DESC = "LGK-weighting 128,batch-aware-dropout=(0.5,0),m=0,noise=pair"
conf_dict = {
	# model configuration
	'embed_aa': True, # True (VHSE) | False | 'learn'
	'gl_pool': 'avg', # both|avg
	'L1_features': 128, # 128|256
	'cl_features': 1024, # classifier hidden neurons count
	'conv_features': 8, # 8 initially, 32 in the paper
	# training configuration
	'warmup_epoch': False,
	'num_epochs_warmup_lr': 0,
	'num_epochs_const_lr': 60, # 100
	'num_epochs_const_lr_2': 150, # 300
	'num_epochs': 400,
	'batch_size': 4, # for gradient calculation
	'learning_rate': 0.001/2,
	'learning_rate_2': 0.0001/2,
	'learning_rate_3': 0.00001,
	'weight_decay': 5e-4,
	'clip_grads': 10.0,
	# data configuration
	'noisy_coords': "pair",
	'norm_size': 4 # 8 (as 'batch_size') # computational batch (influences batch normalization)
}

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.nn import radius_graph as ball_query
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, InstanceNorm, BatchNorm
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.pool import avg_pool_x
from torch_geometric.data import Data
from torch_geometric.transforms import Distance
from torch.nn.functional import one_hot
from torch.nn import Embedding, Linear, Sequential, ReLU, Sigmoid
from torch.nn import Dropout3d as Dropout # Dropout2d, Dropdout, 3d and Dropout1d are calling the same function underneath (the last one available since PyTorch 1.12)
from torch_scatter import scatter
from sklearn.metrics import balanced_accuracy_score as BA_score

from torch.utils.data import DataLoader

from .utils import feed, Feeder, _Ensemble # feed for backward compatibility (import from this module)

EC_CLASSES = 1 # 1 (2) class or regression
AA_CLASSES = 21 # 20 standard AAs + X
VHSE_DIM   = 8  # dimension count of VHSE embedding
CONV_HIDDENS = conf_dict['conv_features']
MAX_HOPS = 6
DROPOUT_RATE = 0 # 0.2
DROPOUT_CL_RATE = 0.5

# some PyTorch Geometric function does not respect batch_mask
def batch_clusters(cluster, batch_mask, safe_margin=2):
	return cluster + (cluster.max() + safe_margin) * batch_mask # +2 so by x_-1/2 and x_0/2 (consecutive points from different examples) never fall into the same cluster

class BatchAwareDropout(torch.nn.Module):
	r"""A placeholder identity operator that is argument-insensitive.

	Args:
		args: any argument (unused)
		kwargs: any keyword argument (unused)

	Shape:
		- Input: :math:`(*)`, where :math:`*` means any number of dimensions.
		- Output: :math:`(*)`, same shape as the input.

	Examples::

		>>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
		>>> input = torch.randn(128, 20)
		>>> output = m(input)
		>>> print(output.size())
		torch.Size([128, 20])

	"""
	def __init__(self, p: float = 0.5) -> None:
		super().__init__()
		self.p = p

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		if(self.p and self.training):
			feature_shape = input[0].shape
			mask = (
					input.new_empty(feature_shape).uniform_() # tensor with uniformly distributed random numbers on the same device as input and dimensions as one data instance
					> self.p                                  # thresholding by dropout rate
				).float().unsqueeze(0) / (1 - self.p)         # normalization to keep similar "weight sum"
			
			input = input.mul(mask)
		return input
Dropout = BatchAwareDropout

class Print(torch.nn.Module):
	def __init__(self):
		super(Print, self).__init__()

	def forward(self, x):
		# print(x[0:20])
		return x

# Double Layer Perceptron
# with droupouts + batch norm. and ReLU for the hidden layer
class DLP(torch.nn.Module):
	def __init__(self, inputs, hiddens, outputs=1): # number of the input, hidden and ouput neurons
		super().__init__()

		self.hid = Sequential(
			Dropout(DROPOUT_CL_RATE),
			Linear(inputs, hiddens),
			BatchNorm(hiddens),
			ReLU()
		)
		self.out = Sequential(
			Dropout(DROPOUT_CL_RATE),
			Linear(hiddens, outputs)
		)

	def forward(self, x):
		# batch norm
		# dropout 0.5
		# relu
		# hidden layer
		x = self.hid(x)
		# batch norm
		# dropout 0.5
		# output layer
		x = self.out(x)
		return x

# Intrinsci-extrinsic convolution layer
class IEConv(torch.nn.Module):
	def __init__(self, inputs, outputs, distance):
		super().__init__()

		self.distance = distance
		self.inputs = inputs   # input features
		self.outputs = outputs # output features
		
		self.intr_dist = Distance(max_value = MAX_HOPS)
		self.extr_dist = Distance(max_value = self.distance)

		self.slp1 = Sequential(
			Linear(2, CONV_HIDDENS), # 2 types of distance
			ReLU()
		)
		self.slp2 = Sequential(
			# Dropout(DROPOUT_CL_RATE),
			Linear(CONV_HIDDENS*inputs, outputs) # largest gradient matrix: [8,I] 
		)
		# effectively implements the following but more frugally in terms of gradient (intermediate) tensor size (8*I << I*O):
		# self.gcl = Sequential(
		#     DLP(2, 8, inputs*outputs), # [8,I*O] matrix size
		#     # ReLU()
		# )

		self.norm = BatchNorm(outputs)
	
	def forward(self,
		graphs: Data, # AAs connected to neighbouring AAs, position: sequential, node features
		coords,       # 3D cartesian coordinates
	):
		neighbors = graphs.edge_index

		# 1st edge feature = intrinsic distance (along bonds)
		graphs = self.intr_dist(graphs) # max_value is used just for nomalization in the step above
		graphs.edge_attr = graphs.edge_attr.clamp(max=1.0) # get values into interval <0,1> for numerical stability
		# NOTE: this way, information about long bond distance is lost (longer than MAX_HOPS)
		
		# 2nd edge feature = extrinsic distance (euclidean)
		graphs.pos = coords
		graphs = self.extr_dist(graphs)

		# batch norm, dropout 0.2, relu

		# get weights from the convolution kernel
		w = self.slp1(graphs.edge_attr)                       # (|edges|, 8)
		w = torch.reshape(w, (-1, 1, CONV_HIDDENS))           # (|edges|, 1, 8)
		# get input features and project them on the edges
		h = graphs.x[neighbors[0]]                            # (|edges|, input_features)
		h = torch.reshape(h, (-1, self.inputs, 1))            # (|edges|, input_features, 1)
		# widen weights 
		h = w*h#torch.matmul(w, h)                            # (|edges|, 8, input_features)
		h = torch.reshape(h, (-1, CONV_HIDDENS*self.inputs))  # (|edges|, 8*input_features)
		assert_test(h)
		# compute the new features factors (per edge)
		# print(h)
		h = self.slp2(h)                                      # (|edges|, output_features)
		assert_test(h)
		# np.savetxt('h_before_scattered.txt', h.detach().cpu().numpy())
		# finish convolution (sum vertex-wise the new features projected on the edges)
		h = scatter(h, neighbors[1], dim=0, dim_size = graphs.num_nodes, reduce='add') # dim_size required - solitary AA may be in PDB (at the end of the sequence)
		# print(h.shape)
		# np.savetxt('h_scattered.txt', h.detach().cpu().numpy())
		assert_test(h)
		h = self.norm(h)
		h = h.relu()

		return h



# like IEConv but employing ResNets
class ResNet(torch.nn.Module):
	def __init__(self, inputs, outputs, distance):
		super().__init__()
		self.distance = distance

		self.ldown = self.SLP(inputs, inputs//4)
		self.conv  = IEConv(inputs//4, inputs, distance)
		self.lup   = self.SLP(inputs, outputs)
		
		self.lside = self.SLP(inputs, outputs) # side channel for passing the features of the node itself

	def forward(self, graph, coords):
		h = graph.x

		graph.x = self.ldown(h)
		x = self.conv(graph, coords)
		x = self.lup(x)
		
		h = self.lside(h)

		return x+h                            # combine features of the node and features of its neighbours
	
	# Single Layer Perceptron with batch norm., dropout and ReLU
	class SLP(torch.nn.Module):
		def __init__(self, inputs, outputs):
			super().__init__()

			self.l = Sequential(
				Print(),
				Dropout(DROPOUT_RATE),
				Print(),
				Linear(inputs, outputs),
				BatchNorm(outputs),
				ReLU()
			)
			self.norm = BatchNorm(outputs)
			
		def forward(self, x):
			# batch norm, dropout 0.2, relu
			# x = x.dropout(DROPOUT_RATE)
			x = self.l(x)
			# x = self.norm(x).relu()
			return x





class PlaNNet(torch.nn.Module):
	"""possible names:
		PCNN – Protein/Peptide/Polyamino-acid Convolutional NN. BUT: "Pulse Coupled NN"
		CCNN - Conformation Convolutional NN. BUT: Constrained Convolutional NN
		ACNN - polyAmino-acid Convolutional NN. BUT: Anatomically Constrained NN
		PLN - Protein Learning (neural) Network
		ACN (AACCNN) - Amino-Acid Chain-Convolutional NN
		NNfP = NN for Proteins
		PLearner = Protein Learner
		PlaNNet /ˈplænet/ = Protein Learning Neural NETwork
	"""
	class EncodeAA:
		def __call__(self, AAs):
			return one_hot(AAs, AA_CLASSES).to(torch.float32)
	class EmbedAA(torch.nn.Module):
		_norm = None
		
		def __init__(self, precomputed: bool = True):
			super().__init__()
			self._precomputed = precomputed
			if precomputed:
				vhse_coeffs = np.genfromtxt("code/VHSE.csv", delimiter=',', skip_header=1, usecols=range(1, VHSE_DIM+1))
				vhse_coeffs = np.vstack([
					vhse_coeffs,
					np.zeros(vhse_coeffs.shape[1]) # 0s as the vector for 'X' AA
				])
				vhse_coeffs = torch.from_numpy(vhse_coeffs)
				self.emb = Embedding.from_pretrained(vhse_coeffs)
			else:
				self.emb = Embedding(AA_CLASSES, VHSE_DIM) # embedding + batch_norm
				self._norm = BatchNorm(VHSE_DIM)
		
		def __call__(self, AAs):
			emb = self.emb(AAs)
			if self._norm:
				self._norm(emb)
			return emb
	
	def __init__(self,
		gl_pool:     str  = conf_dict['gl_pool'],
		embed_aa:    bool = bool(conf_dict['embed_aa']),      # embedding (otherwise 1hot encoding)
		embed_learn: bool = conf_dict['embed_aa'] == 'learn', # learn embedding (or precomputed VHSE)
		L1_features: int  = conf_dict['L1_features'],
		cl_features: int  = conf_dict['cl_features'],
	**_):
		super().__init__()
		
		# MODEL HYPERPARAMETERS
		# hidden layers features
		L1C_FEATURES = L1_features
		L2C_FEATURES = L1C_FEATURES*2
		L3C_FEATURES = L2C_FEATURES*2
		self.LF__FEATURES = L3C_FEATURES + (L3C_FEATURES if gl_pool == 'both' else 0) # avg (+ max)
		
		self._gl_pool = gl_pool
		torch.manual_seed(42)
		
		self.AAenc = self.EmbedAA(not embed_learn) if embed_aa else self.EncodeAA()

		# MODEL LAYERS
		# don't do batch norm, ReLU - parameters
		self.gcl3 = IEConv(VHSE_DIM if embed_aa else AA_CLASSES, L1C_FEATURES, 8)
		# no pooling
		self.gcl3_ = ResNet(L1C_FEATURES, L1C_FEATURES, 8)
		# no pooling
		self.gcl3__ = ResNet(L1C_FEATURES, L1C_FEATURES, 8)
		# pooling
		self.gcl4 = ResNet(L1C_FEATURES, L2C_FEATURES, 12)
		# no pooling
		self.gcl4_ = ResNet(L2C_FEATURES, L2C_FEATURES, 12)
		# pooling
		self.gcl5 = ResNet(L2C_FEATURES, L3C_FEATURES, 16)
		# no pooling
		self.gcl5_ = ResNet(L3C_FEATURES, L3C_FEATURES, 16)
		# pooling

		self.classifier = DLP(self.LF__FEATURES, cl_features, EC_CLASSES)
		
	def forward(self,
		AA_type,
		coordinate,
		seq_position,
		axes,
		batch_mask
	):
		batch_mask = batch_mask.to(torch.int64)
		#print(AA_type)
		AA_type = self.AAenc(AA_type).to(torch.float32)
		# print(AA_type)
		assert_test(AA_type)
		# print(coordinate)
		# print(batch_mask.shape, seq_position.shape, coordinate.shape)
		seq_position = torch.reshape(seq_position.to(torch.float32), (-1,1))
		# print(seq_position.view(-1))
		assert_test(seq_position)

		# 1st convolutional layer (AA level; 8-Å radius)
		# print("ball query:", coordinate, coordinate.size, batch_mask)
		neighbors = ball_query(coordinate, self.gcl3.distance, batch_mask) # [[tos] [froms]], e.g. [to0, to2, ...], [from1, from1, from2, ...]
		# print(neighbors)
		graphs = Data(
			x = AA_type.to(torch.float32),
			edge_index = neighbors,
			pos = seq_position
		)

		assert_test(neighbors)
		assert_test(coordinate)
		# print(AA_type.shape )
		h = self.gcl3(graphs.clone(), coordinate)
		assert_test(h)
		# print(h.shape)
		graphs.x = h
		h = self.gcl3_(graphs.clone(), coordinate)
		# input() # DEBUG
		self.act3 = h
		graphs.x = h
		h = self.gcl3__(graphs, coordinate)
		self.act3 = h
		#print(neighbors)
		#h = self.gconv3(AA_type.to(torch.float32), neighbors)
		# pooling
		clusters = torch.div(seq_position.flatten(), 2, rounding_mode = "trunc")
		#print(clusters)
		clusters = batch_clusters(clusters, batch_mask)
		#print(clusters)
		#print(h.shape, coordinate.shape)
		#print(batch_mask)
		#print(coordinate)
		coordinate, _        = avg_pool_x(clusters, coordinate, batch_mask)
		h, _                 = avg_pool_x(clusters, h, batch_mask)
		clusters, batch_mask = avg_pool_x(clusters, clusters, batch_mask)
		#print(coordinate)
		#print(clusters, batch_mask)
		#print(h.shape, coordinate.shape)

		# 2nd convolutional layer (2 AAs level; 12-Å radius)
		neighbors = ball_query(coordinate, self.gcl4.distance, batch_mask)
		graphs = Data(
			x = h,
			edge_index = neighbors,
			pos = torch.reshape(clusters, (-1,1))
		)

		h = self.gcl4(graphs.clone(), coordinate)
		graphs.x = h
		h = self.gcl4_(graphs, coordinate)
		self.act4 = h
		# h = self.gconv4(h, neighbors)
		clusters = torch.div(clusters, 2, rounding_mode = "trunc")
		# print(clusters)
		clusters = batch_clusters(clusters, batch_mask)
		
		coordinate, _ = avg_pool_x(clusters, coordinate, batch_mask)
		h, batch_mask = avg_pool_x(clusters, h, batch_mask)
		#print(clusters, batch_mask, clusters.shape)
		#print(h.shape)

		# 3rd convolutional layer (4 AAs level; 16-Å radius)
		neighbors = ball_query(coordinate, self.gcl5.distance, batch_mask)
		graphs = Data(
			x = h,
			edge_index = neighbors,
			pos = torch.reshape(clusters, (-1,1))
		)

		h = self.gcl5(graphs.clone(), coordinate)
		graphs.x = h
		h = self.gcl5_(graphs, coordinate)
		self.act5 = h
		# h = self.gconv5(h, neighbors)
		assert_test(h)
		
		# global pooling
		g = global_mean_pool(h, batch_mask)
		if self._gl_pool == "both":
			g2 = global_max_pool(h, batch_mask)
			g = torch.stack([g, g2], 1)
		g = torch.reshape(
			g,
			(-1, self.LF__FEATURES)
		)
		#print(h, h.shape)
		#activations = self.act(h)
		#print(activations)
		assert_test(g)
		# print('cl:', self.classifier(g))

		return self.classifier(g)

class MutPred(torch.nn.Module):
	def __init__(self,
		base_nn: torch.nn.Module
	):
		super().__init__()
		self.base_nn = base_nn
		
	def forward(self,
		AA_type,
		coordinate,
		seq_position,
		axes,
		batch_mask
	):
		base_pred = self.base_nn(AA_type, coordinate, seq_position, axes, batch_mask)
		LOG(base_pred.view(-1), sep='\n')
		pred = base_pred[1::2] - base_pred[0::2] # MUT - WT predictions
		LOG(pred.sigmoid().view(-1))
		return pred.sigmoid()


# log after a keyboard event (CTRL+BREAK on Windows)
class LOG:
	def __init__(self):
		LOG.on = False
		return # TODO: signal.SIGQUIT does not exist on Windows Python 3.8
		signal.signal(signal.SIGQUIT, self.signal_handler) # CTRL+\ on Linux (normally kills the process)
		
	def __call__(self, *args, sep=' '):
		if LOG.on is not False:
			LOG.on = None
			print(*args, sep=sep)
	
	def signal_handler(*args):
		print()
		LOG.on = True
	
	@staticmethod
	def iter():
		LOG.on = not not LOG.on


def assert_test(tensor, mask = None):
	if not mask:
		# l = int(len(tensor) / conf_dict['norm_size']/2)
		l = int(len(tensor) / 2)
		# print(l)
		# equal won't go well with small inaccuracies after ~7 significant digits
		# assert torch.allclose(tensor[0:l], tensor[l:]), (tensor[0:l], tensor[l:])


def Ensemble(paths_or_n): # Ensemble consisting of this-version models
	return _Ensemble(paths_or_n, PlaNNet, MutPred)


# online logging
LOG = LOG()
# LOG.on = True








