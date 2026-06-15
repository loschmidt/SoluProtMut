from . import *

conf_dict['lr_scheduler'] = "ReduceLROnPlateau" # flexible scheduler!
dir = os.path.splitext(__file__)[0] + '/'

models = []
for model in glob.glob("experiments/train-on-DMS-leave-TEM/model*"):
	models.append(restore_model(model, "loss", freeze=False))

TrainMultiple(models, conf_dict, feed, dir, VERSION, VERSION_DESC, "TEM_bymut.train")