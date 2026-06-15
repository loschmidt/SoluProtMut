from . import *

datasets = ["LGK_dupl_bymut.v1.train", "PKS_bymut.v1.train"]
dir = os.path.splitext(__file__)[0] + '/'

TrainMultiple(models, conf_dict, feed, dir, VERSION, VERSION_DESC, datasets)