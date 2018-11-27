import torch as tc
from torch import nn
import torch.nn.functional as F
import Lib.nn_helper as h


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
  def __init__(self,backbone_cfg,IDA_cfg,class_num):
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

    ###########         layers         ####################
    self.input_layer = self._make_input()
    self.stage_1 = self._make_stage(1)
    self.stage_2 = self._make_stage(2)
    self.stage_3 = self._make_stage(3)
    self.stage_4 = self._make_stage(4)

    if IDA_cfg.merge_type==1 or IDA_cfg.merge_type==3:
      self.upsamplers = self._make_stage_upsamplers_dynamic()
      #As we merge output to the last stage, the last output has c[-1] channels.
      self.last_out_channel = self.c[-1]
    elif IDA_cfg.merge_type==2 or IDA_cfg.merge_type==4:
      self.upsamplers = self._make_stage_upsamplers_fixed(IDA_cfg.fixed_merge_channels)
      #As we merge output to have the fixed number of channels, the last output has IDA_cfg.fixed_merge_channels channels.
      self.last_out_channel = IDA_cfg.fixed_merge_channels
    else:
      raise Exception('IDA config error, merge_type should be one of [1,2,3,4]')

    self.upsampler_2 = self.upsamplers[2]
    self.upsampler_3 = self.upsamplers[3]
    self.upsampler_4 = self.upsamplers[4]

    self.stage_bns = self._make_stage_bns()
    self.stage_1_bn = self.stage_bns[1]
    self.stage_2_bn = self.stage_bns[2]
    self.stage_3_bn = self.stage_bns[3]
    self.stage_4_bn = self.stage_bns[4]

    self.logits_layer = self._make_semantic_output(number_class = class_num)
    self.relu = nn.ReLU(inplace=True)

  def _make_input(self,shrinkage= 0.5, group = 32):
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

  def _make_stage_bns(self):
    if self.IDA_cfg.merge_type==1 or self.IDA_cfg.merge_type==3:
      return {1:nn.BatchNorm2d(c_1),2:nn.BatchNorm2d(c_2),3:nn.BatchNorm2d(c_3),4:nn.BatchNorm2d(c_4)}
    elif self.IDA_cfg.merge_type==2 or self.IDA_cfg.merge_type==4:
      c = self.IDA_cfg.fixed_merge_channels
      return {1:nn.BatchNorm2d(c),2:nn.BatchNorm2d(c),3:nn.BatchNorm2d(c),4:nn.BatchNorm2d(c)}

  def _make_stage_upsamplers_dynamic(self):
    ''' upsample output from stage_idx to fit output from stage_idx-1 '''
    upsamplers = {}
    for i in range(2,self.num_stages+1):
      c_in = self.c[i-1]
      c_out = self.c[i]
      conv = h.conv1x1(c_in,c_out)
      deconv = h.deconv_upsample(c_out,c_out,upsample_rate = 2,group = 1)
      upsamplers[i] = nn.Sequential(conv,deconv)
    return upsamplers

  def _make_stage_upsamplers_fixed(self,c_in_out=32):
    ''' upsample output from stage_idx to fit output from stage_idx-1 '''
    upsamplers = {}
    for i in range(2,self.num_stages):
      c_in = c_in_out
      c_out = c_in_out
      conv = h.conv1x1(c_in,c_out)
      deconv = h.deconv_upsample(c_out,c_out,upsample_rate = 2,group = 1)
      upsamplers[i] = nn.Sequential(conv,deconv)
    return upsamplers

  def _make_semantic_output(self,number_class):
    c_in = self.last_out_channel
    return conv1x1(c_in, number_class)

  '''
  4 different types of merge strategy.
  1) Adaptive channel merge      : normal addition, keep channels of each stage.
  2) Fixed channel merge         : normal addition, keep same channels(eg.32 in the original paper) for each stage.
  3) Gated adaptive channel merge: gated normal addition, keep channels of each stage.
  4) Gated fixed channel merge   : gated normal addition, keep same channels(eg.32 in the original paper) for each stage.
  '''
  def _forward_IDA(self,x1,x2):
    '''
    Nomral addition, keep channels of each stage.
    Args:
      x1: lower level tensor
      x2: higher level tensor
    '''
    if self.IDA_cfg.merge_type==1 or self.IDA_cfg.merge_type==2:
      return self._forward_IDA_merge(x1,x2)
    elif self.IDA_cfg.merge_type==3 or self.IDA_cfg.merge_type==4:
      return self._forward_IDA_gated_merge(x1,x2)
    else:
      raise Exception('merge type has to be 1,2,3 or 4')

  def _forward_IDA_merge(self,x1,x2,stage_idx):
    '''
    Args:
      x1: lower level tensor
      x2: higher level tensor
    '''
    out = x1+self.upsamplers[stage_idx](x2)
    return out

  def _forward_IDA_gated_merge(self,x1,x2,stage_idx):
    '''
    Args:
      x1: lower level tensor
      x2: higher level tensor
    '''
    pass
    return out

  def forward(self,x):
    #basic stages
    outInput = self.input_layer(x)
    outStg1 = self.stage_1(outInput)
    outStg2 = self.stage_2(outStg1)
    outStg3 = self.stage_3(outStg2)
    outStg4 = self.stage_4(outStg3)
    #IDA stages
    #level 1
    outStg2_L1_tmp = self.upsampler_2(outStg2)+outStg1
    outStg2_L1_tmp = self.stage_1_bn(outStg2_L1_tmp)
    outStg2_L1 = self.relu(outStg2_L1_tmp)

    outStg3_L1_tmp = self.upsampler_3(outStg3)+outStg2
    outStg3_L1_tmp = self.stage_2_bn(outStg3_L1_tmp)
    outStg3_L1 = self.relu(outStg3_L1_tmp)

    outStg4_L1_tmp = self.upsampler_4(outStg4)+outStg3
    outStg4_L1_tmp = self.stage_3_bn(outStg4_L1_tmp)
    outStg4_L1 = self.relu(outStg4_L1_tmp)

    #level 2
    outStg3_L2_tmp = self.upsampler_3(outStg3_L1)+outStg2_L1
    outStg3_L2_tmp = self.stage_2_bn(outStg3_L2_tmp)
    outStg3_L2 = self.relu(outStg3_L2_tmp)

    outStg4_L2_tmp = self.upsampler_4(outStg4_L1)+outStg3_L1
    outStg4_L2_tmp = self.stage_3_bn(outStg4_L2_tmp)
    outStg4_L2 = self.relu(outStg4_L2_tmp)

    #level 3
    outStg4_L3_tmp = self.upsampler_4(outStg4_L2)+outStg3_L2
    outStg4_L3 = self.stage_3_bn(outStg4_L3_tmp)

    out = self.logits_layer(outStg4_L3)


  