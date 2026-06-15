from . import *

dir = os.path.splitext(__file__)[0] + '/'
TrainMultiple(models, conf_dict, feed, dir, VERSION, VERSION_DESC, "TEM_bypos.v1.train")