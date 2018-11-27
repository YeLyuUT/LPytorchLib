import torch as tc
from torch import nn
import torch.nn.functional as F
import Lib.nn_helper as h


class Module(nn.Module):
  def __init__(self):
    super(Module,self).__init__()
    #self.



class ResidualBlock(nn.Module):
  '''This class implement ResNeXtBlock'''

  '''
  shrinkage: downsample channel number shrinkage factor relative to c_out, range 0 - 1 exclusively.
  expansion: output channel number expansion factor(relative to block input), can be 2 or 4 or other positive integer number.
  '''
  def __init__(self, c_in ,c_out, shrinkage= 0.5, group = 32, downsample = False, final_relu = True):
    super(ResidualBlock,self).__init__()
    self.final_relu = final_relu

    if downsample is True:
      shrinkage*=2

    c_mid = c_out*shrinkage

    self.conv1 = h.conv1x1(c_in,c_mid)
    self.bn1 = nn.BatchNorm2d(c_mid)

    if downsample is False:
       self.conv2 = h.conv3x3(c_mid,c_mid,group = group)
       self.conv_skip = None
    else:
       self.conv2 = h.conv3x3(c_mid,c_mid,stride = 2,group = group)
       self.conv_skip = h.conv3x3(c_in,c_out,stride = 2,group = group)

    self.bn2 = nn.BatchNorm2d(c_mid)
    self.conv3 = h.conv1x1(c_mid, c_out)
    self.relu = nn.ReLU(inplace=True)

  def forward(self,x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.conv_skip is not None:
      residual = self.conv_skip(x)

    out+=residual
    if self.final_relu is True:
      out = self.relu(out)

    return out

class ResNeXt_IDA(nn.Module):
  def __init__(self,backbone_cfg,IDA_cfg):
    super(ResNeXt_IDA,self).__init__()
    self.backbone_cfg = backbone_cfg
    self.IDA_cfg = IDA_cfg
    self.input_block_in_channels = backbone_cfg.input_block_in_channels
    self.channel_nums = backbone_cfg.channelNums
    self.blockNums = backbone_cfg.blockNums

    assert(len(self.channel_nums)==len(self.blockNums))

    self.c_0 = self.input_block_in_channels
    self.c_1 = self.channel_nums[0]
    self.c_2 = self.channel_nums[1]
    self.c_3 = self.channel_nums[2]
    self.c_4 = self.channel_nums[3]
    self.c = [self.c_0,self.c_1,self.c_2,self.c_3,self.c_4]

    self.num_stages = len(self.channel_nums)

  def _make_input(self,shrinkage= 0.5, group = 32, downsample = False):
    conv1 = h.conv2d(3,c_out=self.c_0, group=1, stride=2, kernel_size=7,padding=3)
    bn1 = nn.BatchNorm2d(self.c_0)
    relu = nn.ReLU(inplace = True)
    maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
    layers = []
    layers.extend([conv1,bn1,relu,maxpool])
    return nn.Sequential(*layers)

  def _make_stage(self,stage_idx, shrinkage= 0.5, group = 32):
    assert(stage_idx>0 and stage_idx<self.num_stages)
    layers = []
    c_in = c[stage_idx-1]
    c_out = c[stage_idx]
    for idx in xrange(self.blockNums(stage_idx)):
      block = None
      #if downsample block
      if idx ==0:
        block = self.ResidualBlock(c_in = c_in , c_out = c_out, shrinkage= shrinkage, group = group, downsample = True)
      #else normal block
      else:
        block = self.ResidualBlock(c_in = c_out , c_out = c_out, shrinkage= shrinkage, group = group, downsample = False)
      layers.append(block)
    return nn.Sequential(*layers)

  '''
  4 different types of merge strategy.
  1) Adaptive channel merge      : normal addition, keep channels of each stage.
  2) Fixed channel merge         : normal addition, keep same channels(eg.32 in the original paper) for each stage.
  3) Gated adaptive channel merge: gated normal addition, keep channels of each stage.
  4) Gated fixed channel merge   : gated normal addition, keep same channels(eg.32 in the original paper) for each stage.
  '''
  def _forward_IDA_merge(self,x1,x2,merge_type = 2):
    '''
    Nomral addition, keep channels of each stage.
    Args:
      x1: lower level tensor
      x2: higher level tensor
    '''
    if merge_type==1:
      return self._forward_IDA_adaptive_merge(x1,x2)
    elif merge_type==2:
      return self._forward_IDA_fixed_merge(x1,x2)
    elif merge_type==3:
      return self._forward_IDA_gated_adaptive_merge(x1,x2)
    elif merge_type==4:
      return self._forward_IDA_gated_fixed_merge(x1,x2)
    else:
      raise Exception('merge type has to be 1,2,3 or 4')
    return out

  def _forward_IDA_fixed_merge(self,x1,x2):
    '''
    Nomral addition, keep channels of each stage.
    Args:
      x1: lower level tensor
      x2: higher level tensor
    '''
    x2 = 
    return out

  def _forward_IDA_adaptive_merge(self,x1,x2):
    '''
    Nomral addition, keep channels of each stage.
    Args:
      x1: lower level tensor
      x2: higher level tensor
    '''
    pass
    return out

  def _forward_IDA_gated_fixed_merge(self,x1,x2):
    '''
    Nomral addition, keep channels of each stage.
    Args:
      x1: lower level tensor
      x2: higher level tensor
    '''
    pass
    return out

  def _forward_IDA_gated_adaptive_merge(self,x1,x2):
    '''
    Nomral addition, keep channels of each stage.
    Args:
      x1: lower level tensor
      x2: higher level tensor
    '''
    pass
    return out


  