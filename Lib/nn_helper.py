import torch as tc

__all__=['conv2d','conv3x3','conv1x1']

def conv2d(c_in,c_out, group, stride, kernel_size,padding,bias = False):
 return tc.nn.Conv2d(c_in, c_out, kernel_size = kernel_size, stride = stride, padding = padding, group = group, bias = bias)

def conv3x3(c_in,c_out,  stride = 1,padding = 1,group = 1,bias = False):
 return conv2d(c_in,c_out , stride = stride, kernel_size = 3,padding = 1, group = group, bias = bias)

def conv1x1(c_in,c_out,group = 1,bias = False):
 return conv2d(c_in,c_out , stride = 1, kernel_size = 1,padding = 0, group = group, bias = bias)

def deconv_upsample(c_in,c_out,upsample_rate = 2):
  ''' deconvolution for upsampling, 
      upsample_rate controls the upsampling rate, can be 2, 4 , 8 or 16.
      if upsample_rate==2, then the tensor width and height are 2x of the original size.
  '''
  stride = 1*upsample_rate
  kernel_size = 2*upsample_rate
  #TODO
  tc.nn.ConvTranspose2d(c_in,c_out,kernel_size=kernel_size,
    stride = stride,padding=0, output_padding=0, groups=1, bias=True, dilation=1)