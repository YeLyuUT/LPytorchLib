import os
import os.path as osp

import torch.optim as optim
import torch.nn as nn

from Model.IDA import 
from Model.model_cfg import cfg_ResNeXt101,cfg_IDA
from Lib.weight_init import normal_init
from Lib.model_saver import Flag_save_load
from proj_config import cfg


opt={'model_saving_dir':cfg.model_saving_dir,
          'ckpt_prefix':None}

if __name__=='__main__':
  device = torch.device("cuda:0" if opt.cuda else "cpu")
  model = ResNeXt_IDA(cfg_ResNeXt101,cfg_IDA).to(device)
  model.apply(normal_init)

  if osp.exists(opt['ckpt_prefix']):
    ckpt_saver = PytorchCkpt()
    ckpt_saver.load(prefix,model, load_flag = Flag_save_load.LG, optimizer = None,train_mode = True,suffix='')

  #TODO write loss
  loss_op = nn.NLLLoss(ignore_index=-100)

