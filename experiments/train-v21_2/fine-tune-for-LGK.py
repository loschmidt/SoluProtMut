from . import *

conf_dict['lr_scheduler'] = "ReduceLROnPlateau" # flexible scheduler!
dir = os.path.splitext(__file__)[0] + '/'

models = []
for model in glob.glob("experiments/train-v21_2/on-DMS-leave-LGK/model*"):
	models.append(restore_model(model, "loss", freeze=False))

TrainMultiple(models, conf_dict, feed, dir, VERSION, VERSION_DESC, "LGK_dupl_bymut.v1.train")