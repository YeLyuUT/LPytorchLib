#This script contains the code for model saving, only state_dict is saved.
from os import path as osp
from enum import Enum
import torch

class Flag_save_load(Enum):
  #Save on CPU
  SC = 0
  #Save on GPU
  SG = 1
  #Load on CPU
  LC = 2
  #Load on GPU
  LG = 3

class PytorchCkpt:
  @property
  def model(self):
    return self._model
  @property
  def optimizer(self):
    return self._optimizer
  @property
  def epoch(self):
    return self._epoch
  @property
  def save_flag(self):
    return self._save_flag

  def __init__(self):
    self._reset()
  def __getitem__(self,key):
    return self.dict[key]
  def __setitem__(self,key,value):
    self.dict[key] = value
  def __call__(self):
    return self.dict
  def _reset(self):
    self._model = None
    self._optimizer = None
    self._epoch = None
    self._save_flag = Flag_save_load.SG

  def _get_model_path(self,prefix,suffix):
    save_path = osp.join(prefix,'model'+suffix+'.pt')
    return save_path

  def _get_optimizer_path(self,prefix,suffix):
    save_path = osp.join(prefix,'optimizer'+suffix+'.pt')
    return save_path

  #Path for other non-parameter variables
  def _get_addition_path(self,prefix,suffix):
    save_path = osp.join(prefix,'addition'+suffix+'.pd')
    return save_path

  #A common PyTorch convention is to save models using either a .pt or .pth file extension
  def _save_model(self,path, model):
    torch.save(model.state_dict(), path)


  def _load_model(self,path,model,save_flag,load_flag,train_mode = False):
    '''load model parameters
    Arg:
        train_mode:whether to load model for training
    '''
    if save_flag is Flag_save_load.SG and load_flag is Flag_save_load.LC:
      device = torch.device('cpu')
      model.load_state_dict(torch.load(path, map_location=device))
    elif save_flag is Flag_save_load.SG and load_flag is Flag_save_load.LG:
      device = torch.device("cuda")
      model.load_state_dict(torch.load(path))
      model.to(device)
    elif save_flag is Flag_save_load.SC and load_flag is Flag_save_load.LG:
      device = torch.device("cuda")
      model.load_state_dict(torch.load(path, map_location="cuda:0"))
      model.to(device)
    else:
      model.load_state_dict(torch.load(path))

    if train_mode is True:
      model.train()
    else:
      model.eval()

    return model

  #A common PyTorch convention is to save models using either a .pt or .pth file extension
  def save(self,prefix,model,save_flag,optimizer=None,epoch=None,suffix = ''):
    assert(isinstance(save_flag,Flag_save_load))
    assert(save_flag is Flag_save_load.SC or save_flag is Flag_save_load.SG)
    #save model
    save_path = self._get_model_path(prefix,suffix)
    self._save_model(save_path, model)
    #save optimizer
    if optimizer is not None:
      save_path = self._get_optimizer_path(prefix,suffix)
      self._save_model(save_path, optimizer)
    #save other non-parameter variables
    save_path = self._get_addition_path(prefix,suffix)
    addition_dict = {'epoch':self.epoch,'save_flag':self.save_flag}
    torch.save(addition_dict,save_path)

  def load(self,prefix,model,load_flag, optimizer = None,train_mode = True,suffix=''):
    assert(isinstance(load_flag,Flag_save_load))
    assert(load_flag is Flag_save_load.LC or load_flag is Flag_save_load.LG)
    self._reset()
    #load other non-parameter variables
    load_path = self._get_addition_path(prefix,suffix)
    addition_dict = torch.load(load_path)
    self._save_flag = addition_dict['save_flag']
    self._epoch = addition_dict['epoch']
    #load model
    load_path = self._get_model_path(prefix,suffix)
    self._model = self._load_model(load_path,model,self._save_flag,load_flag,train_mode = train_mode)
    #load optimizer
    if optimizer is not None:
      load_path = self._get_optimizer_path(prefix,suffix)
      self._optimizer = self._load_model(load_path,optimizer,self._save_flag,load_flag,train_mode = train_mode)

    return ckpt

    '''
if __name__=='__main__':
  ckpt = PytorchCkpt()
  ckpt['model_state_dict'] = 1
  ckpt['optimizer_state_dict'] = 2
  ckpt['epoch'] = 3
  print(ckpt())
  '''

