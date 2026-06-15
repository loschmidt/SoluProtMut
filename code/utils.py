import sys, platform, torch, torch_geometric

DEVICE_DUMP = {}
DEVICE_DUMP['node_name'] = platform.node()
DEVICE_DUMP['cuda.dev'] = torch.cuda.get_device_name()
DEVICE_DUMP['cuda.count'] = torch.cuda.device_count()
DEVICE_DUMP['ver.cuda'] = torch.version.cuda # the CUDA SDK version, not runtime/driver
DEVICE_DUMP['ver.torch'] = torch.__version__
DEVICE_DUMP['ver.pyg'] = torch_geometric.__version__
DEVICE_DUMP['ver.python'] = sys.version

def dump_device(file):
	alignment = max([len(key) for key in DEVICE_DUMP])
	
	for item, val in DEVICE_DUMP.items():
		print(item +':', " "*(alignment - len(item)) + str(val), file=file)
	# e.g.:
	# node_name:  ani1.rcxsc.loschmidt.cz
	# cuda.dev:   NVIDIA GeForce GTX 1080 Ti
	# cuda.count: 1
	# ver.cuda:   11.3
	# ver.torch:  1.10.0+cu113
	# ver.pyg:    2.0.4
	# ver.python: 3.8.13 (default, Mar 28 2022, 11:38:47)
