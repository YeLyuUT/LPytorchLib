from easydict import EasyDict as edict
#########################    
#    backbone config    #
#########################
#ResNeXt50 config
cfg_ResNeXt50 = edict()
cfg_ResNeXt50.blockNums = [3,4,6,3]
cfg_ResNeXt50.channelNums = [256,512,1024,2048]
cfg_ResNeXt50.input_block_in_channels = 64

#ResNeXt101 config
cfg_ResNeXt101 = edict()
cfg_ResNeXt101.blockNums = [3,4,23,3]
cfg_ResNeXt101.channelNums = [256,512,1024,2048]
cfg_ResNeXt101.input_block_in_channels = 64

#ResNeXt152 config
cfg_ResNeXt152 = edict()
cfg_ResNeXt152.blockNums = [3,4,36,3]
cfg_ResNeXt152.channelNums = [256,512,1024,2048]
cfg_ResNeXt152.input_block_in_channels = 64

#########################    
#      IDA  config      #
#########################
#IDA config
cfg_IDA = edict()

#1) Adaptive channel merge      : normal addition, keep channels of each stage.
#2) Fixed channel merge         : normal addition, keep same channels(eg.32 in the original paper) for each stage.
#3) Gated adaptive channel merge: gated normal addition, keep channels of each stage.
#4) Gated fixed channel merge   : gated normal addition, keep same channels(eg.32 in the original paper) for each stage.
cfg_IDA.merge_type = 2#should be 1 or 2 or 3 or 4

cfg_IDA.fixed_merge_channels = 32#same as the original paper

