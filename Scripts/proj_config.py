import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

_C = edict()

cfg = _C

cfg.cache_dir=None

#model saving directory
cfg.model_saving_path = '/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Work/PytorchLib/Data/saved_model'


