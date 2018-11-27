from easydict import EasyDict as edict

#ResNeXt50 config
ResNeXt50 = edict()
ResNeXt50.blockNums = [3,4,6,3]
ResNeXt50.channelNums = [256,512,1024,2048]
ResNeXt50.input_block_in_channels = 64

#ResNeXt101 config
ResNeXt101 = edict()
ResNeXt101.blockNums = [3,4,23,3]
ResNeXt101.channelNums = [256,512,1024,2048]
ResNeXt101.input_block_in_channels = 64

#ResNeXt152 config
ResNeXt152 = edict()
ResNeXt152.blockNums = [3,4,36,3]
ResNeXt152.channelNums = [256,512,1024,2048]
ResNeXt152.input_block_in_channels = 64

#IDA config
IDA = edict()
IDA.fixed_merge_channels = 32#same as the original paper
