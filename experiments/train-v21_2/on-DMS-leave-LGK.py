from . import *

datasets = ["PKS_bymut.v1.train", "TEM_bymut.train"]
dir = os.path.splitext(__file__)[0] + '/'

TrainMultiple(models, conf_dict, feed, dir, VERSION, VERSION_DESC, datasets)