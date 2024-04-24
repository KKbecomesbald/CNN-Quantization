from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from function import FakeQuantize, interp
import numpy as np

def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    qmin = 0
    qmax = 2. ** num_bits - 1
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmax - max_val / scale

    if zero_point < qmin:
        zero_point = qmin
    elif zero_point > qmax:
        zero_point = qmax
    
    zero_point = int(zero_point)

    return scale, zero_point

def quantize_tensor(x, scale, zero_point, num_bits=8, signed=False):
    if signed:
        qmin = -2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0
        qmax = 2. ** num_bits - 1

    q_x = x / scale + zero_point
    q_x.clamp_(qmin, qmax).round_()

    return q_x.float()

def dequantize(q_x, scale, zero_point):
    return scale * (q_x - zero_point)

class QParam:
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.scale = None
        self.zero_point = None
        self.min = None
        self.max = None
    
    def update(self, tensor):
        if self.max is None or self.max < tensor.max():
            self.max = tensor.max()
        
        if self.min is None or self.min > tensor.min():
            self.min = tensor.min()
        
        self.scale, self.zero_point = calcScaleZeroPoint(self.min, self.max, self.num_bits)

    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.scale, self.zero_point, num_bits=self.num_bits)
    
    def dequantize_tensor(self, q_x):
        return dequantize(q_x, self.scale, self.zero_point)
    

class QModule(nn.Module):

    def __init__(self, qi=True, qo=True, num_bits=8):
        super(QModule, self).__init__()
        #qi: quantized input qo: quantized output
        self.num_bits = num_bits
        self.qw = QParam(num_bits=num_bits)
        if qi:
            self.qi = QParam(num_bits=num_bits)
        if qo:
            self.qo = QParam(num_bits=num_bits)
            
    def forward():
        pass

    def freeze():
        pass

    def quantize_inference(self, x):
        raise NotImplementedError('quantize_inference should be implemented!')

class QAdd(nn.Module):
    def __init__(self, qo=True, num_bits=8):
        super(QAdd, self).__init__()
        self.num_bits = num_bits
        self.register_buffer('M1', torch.tensor(0, requires_grad=False))
        self.register_buffer('M2', torch.tensor(0, requires_grad=False))
        if qo:
            self.qo = QParam(num_bits=num_bits)

    def freeze(self, qi_1=None, qi_2=None, qo=None):
        if hasattr(self, 'qi_1') and qi_1 is not None:
            raise ValueError('qi_1 has been provided in init function')
        if not hasattr(self, 'qi_1') and qi_1 is None:
            raise ValueError('qi_1 is not existed, should be provided')
        if hasattr(self, 'qi_2') and qi_2 is not None:
            raise ValueError('qi_2 has been provided in init function')
        if not hasattr(self, 'qi_2') and qi_2 is None:
            raise ValueError('qi_2 is not existed, should be provided')
        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided')

        if qi_1 is not None:
            self.qi_1 = qi_1
            print("qi_1 max is", self.qi_1.max)
        if qi_2 is not None:
            self.qi_2 = qi_2
            print("qi_2 max is", self.qi_2.max)
        if qo is not None:
            self.qo = qo

        self.M1.data = (self.qi_1.scale / self.qo.scale).data
        self.M2.data = (self.qi_2.scale / self.qo.scale).data

    def forward(self, x1, x2):
        x = x1 + x2      
        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)
        print("x1 + x2 max is", self.qo.max)
        print("x1 + x2 min is", self.qo.min)  
        return x

    def quantize_inference(self, x1, x2):
        x1 = x1 - self.qi_1.zero_point
        x2 = x2 - self.qi_2.zero_point
        x = self.M1 * x1 + self.M2 * x2
        x.round_()
        x = x + self.qo.zero_point
        x.clamp_(0., 2.**self.num_bits - 1).round_()
        return x

class QConv2d(QModule):

    def __init__(self, conv_module, qi=True, qo=True, num_bits=8):
        super(QConv2d, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.conv_module = conv_module
        self.qw = QParam(num_bits=num_bits)
        self.register_buffer('M', torch.tensor(0, requires_grad=False))

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided')
        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided')
        
        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        self.M.data = (self.qw.scale * self.qi.scale / self.qo.scale).data
        self.conv_module.weight.data = self.qw.quantize_tensor(self.conv_module.weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point
        
        self.conv_module.bias.data = quantize_tensor(self.conv_module.bias.data, scale=self.qi.scale * self.qw.scale, zero_point=0, signed=True)

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x) 
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.conv_module.weight.data)

        x = F.conv2d(
            input=x, 
            weight=FakeQuantize.apply(self.conv_module.weight, self.qw), 
            bias=self.conv_module.bias,
            stride=self.conv_module.stride,
            padding=self.conv_module.padding,
            dilation=self.conv_module.dilation,
            groups=self.conv_module.groups
        )

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x
    
    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.conv_module(x)
        x = self.M * x
        x.round_()
        x = x + self.qo.zero_point
        x.clamp_(0., 2.**self.num_bits - 1).round_()
        return x

class QConvBNReLU(QModule):

    def __init__(self, conv_module, bn_module, qi=True, qo=True, num_bits=8):
        super(QConvBNReLU, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.training = True
        self.conv_module = conv_module
        self.bn_module = bn_module
        self.qw = QParam(num_bits=num_bits)
        self.qb = QParam(num_bits=32)
        self.register_buffer('M', torch.tensor(0, requires_grad=False))

    def fold_bn(self, mean, std):
        if self.bn_module.affine:
            gamma_ = self.bn_module.weight / std
            weight = self.conv_module.weight * gamma_.view(self.conv_module.out_channels, 1, 1, 1)
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean + self.bn_module.bias
            else:
                bias = self.bn_module.bias - gamma_ * mean
        else:
            gamma_ = 1 / std
            weight = self.conv_module.weight * gamma_
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean
            else:
                bias = -gamma_ * mean
        
        return weight, bias

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)
        if self.training:
            y = F.conv2d(x, self.conv_module.weight, self.conv_module.bias,
                         stride=self.conv_module.stride,
                         padding=self.conv_module.padding,
                         dilation=self.conv_module.dilation,
                         groups=self.conv_module.groups)
            y = y.permute(1, 0, 2, 3) # NCHW -> CNHW
            y = y.contiguous().view(self.conv_module.out_channels, -1)
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            self.bn_module.running_mean = \
                (1 - self.bn_module.momentum) * self.bn_module.running_mean + \
                self.bn_module.momentum * mean
            self.bn_module.running_var = \
                (1 - self.bn_module.momentum) * self.bn_module.running_var + \
                self.bn_module.momentum * var
        else:
            mean  = Variable(self.bn_module.running_mean)
            var = Variable(self.bn_module.running_var)
        std = torch.sqrt(var + self.bn_module.eps)
        weight, bias = self.fold_bn(mean, std)
        self.qw.update(weight.data)
        x = F.conv2d(input=x, 
                     weight=FakeQuantize.apply(weight, self.qw), bias=bias, 
                     stride=self.conv_module.stride, 
                     padding=self.conv_module.padding, 
                     dilation=self.conv_module.dilation, 
                     groups=self.conv_module.groups)
        x = F.relu(x)
        
        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)
        
        return x

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided')
        
        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M.data = (self.qw.scale * self.qi.scale / self.qo.scale).data
        std = torch.sqrt(self.bn_module.running_var +self.bn_module.eps)

        weight, bias = self.fold_bn(self.bn_module.running_mean, std)
        self.conv_module.weight.data = self.qw.quantize_tensor(weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point
        self.conv_module.bias.data = quantize_tensor(bias, scale=self.qi.scale * self.qw.scale,
                                                     zero_point=0, num_bits=32, signed=True)
        
    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.conv_module(x)
        x = self.M * x
        x.round_()
        x = x + self.qo.zero_point
        x.clamp_(0., 2.**self.num_bits-1.).round_()
        return x

class QConvBN(QModule):

    def __init__(self, conv_module, bn_module, qi=True, qo=True, num_bits=8):
        super(QConvBN, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.conv_module = conv_module
        self.bn_module = bn_module
        self.qw = QParam(num_bits=num_bits)
        self.qb = QParam(num_bits=32)
        self.training = True
        self.register_buffer('M', torch.tensor(0, requires_grad=False))

    def fold_bn(self, mean, std):
        if self.bn_module.affine:
            gamma_ = self.bn_module.weight / std
            weight = self.conv_module.weight * gamma_.view(self.conv_module.out_channels, 1, 1, 1)
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean + self.bn_module.bias
            else:
                bias = self.bn_module.bias - gamma_ * mean
        else:
            gamma_ = 1 / std
            weight = self.conv_module.weight * gamma_
            if self.conv_module.bias is not None:
                bias = gamma_ * self.conv_module.bias - gamma_ * mean
            else:
                bias = -gamma_ * mean
        
        return weight, bias

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        if self.training:
            y = F.conv2d(x, self.conv_module.weight, self.conv_module.bias,
                         stride=self.conv_module.stride,
                         padding=self.conv_module.padding,
                         dilation=self.conv_module.dilation,
                         groups=self.conv_module.groups)
            y = y.permute(1, 0, 2, 3) # NCHW -> CNHW
            y = y.contiguous().view(self.conv_module.out_channels, -1)
            mean = y.mean(1).detach()
            var = y.var(1).detach()
            self.bn_module.running_mean = \
                (1 - self.bn_module.momentum) * self.bn_module.running_mean + \
                self.bn_module.momentum * mean
            self.bn_module.running_var = \
                (1 - self.bn_module.momentum) * self.bn_module.running_var + \
                self.bn_module.momentum * var
        else:
            mean  = Variable(self.bn_module.running_mean)
            var = Variable(self.bn_module.running_var)
        std = torch.sqrt(var + self.bn_module.eps)
        weight, bias = self.fold_bn(mean, std)
        self.qw.update(weight.data)  
        x = F.conv2d(input=x, 
                     weight=FakeQuantize.apply(weight, self.qw), bias=bias, 
                     stride=self.conv_module.stride, 
                     padding=self.conv_module.padding, 
                     dilation=self.conv_module.dilation, 
                     groups=self.conv_module.groups)
        
        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)
        
        return x

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided')
        
        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M.data = (self.qw.scale * self.qi.scale / self.qo.scale).data
        std = torch.sqrt(self.bn_module.running_var +self.bn_module.eps)

        weight, bias = self.fold_bn(self.bn_module.running_mean, std)
        self.conv_module.weight.data = self.qw.quantize_tensor(weight.data)
        self.conv_module.weight.data = self.conv_module.weight.data - self.qw.zero_point
        self.conv_module.bias.data = quantize_tensor(bias, scale=self.qi.scale * self.qw.scale,
                                                     zero_point=0, num_bits=32, signed=True)
        
    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.conv_module(x)
        x = self.M * x
        x.round_()
        x = x + self.qo.zero_point
        x.clamp_(0., 2.**self.num_bits-1.).round_()
        return x

class QLinear(QModule):

    def __init__(self, fc_module, qi=True, qo=True, num_bits=8):
        super(QLinear, self).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.fc_module = fc_module
        self.qw = QParam(num_bits=num_bits)
        self.register_buffer('M', torch.tensor(0, requires_grad=False))

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided')
        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided')
        
        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        
        self.M.data = (self.qw.scale * self.qi.scale / self.qo.scale).data

        self.fc_module.weight.data = self.qw.quantize_tensor(self.fc_module.weight.data)
        self.fc_module.weight.data = self.fc_module.weight.data - self.qw.zero_point
        self.fc_module.bias.data = quantize_tensor(
            self.fc_module.bias.data, 
            scale=self.qi.scale * self.qw.scale,
            zero_point=0, 
            num_bits=32, 
            signed=True
        )

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        self.qw.update(self.fc_module.weight.data)

        x = F.linear(
            input=x, 
            weight=FakeQuantize.apply(self.fc_module.weight, self.qw), 
            bias=self.fc_module.bias
        )

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x

    def quantize_inference(self, x):
        x = x - self.qi.zero_point
        x = self.fc_module(x)
        x = self.M * x
        x.round_() 
        x = x + self.qo.zero_point
        x.clamp_(0., 2.**self.num_bits-1.).round_()
        return x

class QLeakyRelu(QModule):
    def __init__(self, negative_slope, qi=True, qo=True, num_bits=8):
        super(QLeakyRelu, self).__init__(qi=qi, num_bits=num_bits)
        self.num_bits = num_bits
        self.negative_slope = negative_slope

    def freeze(self, qi=None):
        
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided')
        
        if qi is not None:
            self.qi = qi
    
    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)
        
        x = F.leaky_relu(x, self.negative_slope)
        return x
    
    def quantize_inference(self, x):
        x = x.clone()
        x[x < self.qi.zero_point] = self.negative_slope * x + self.qi.zero_point * (1 - self.negative_slope)
        x.clamp_(0., 2.**self.num_bits-1.).round_()
        return x


class QRelu(QModule):
    
    def __init__(self, qi=False, num_bits=None):
        super(QRelu, self).__init__(qi=qi, num_bits=num_bits)
        self.num_bits = num_bits

    def freeze(self, qi=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init funciton')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided')
        
        if qi is not None:
            self.qi = qi
    
    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)
        
        x = F.relu(x)
        return x
    
    def quantize_inference(self, x):
        x = x.clone()
        x[x < self.qi.zero_point] = self.qi.zero_point
        return x

class QSigmoid(QModule):
    def __init__(self, qi=True, qo=True, num_bits=8, lut_size=64):
        super(QSigmoid).__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.num_bits = num_bits
        self.lut_size = lut_size
    
    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)
        
        x = torch.sigmoid(x)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x
    
    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided')
        
        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init funciton')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        
        lut_qx = torch.tensor(np.linspace(0, 2 ** self.num_bits - 1, self.lut_size), dtype=torch.uint8)
        lut_x = self.qi.dequantize_tensor(lut_qx)
        lut_y = torch.sigmoid(lut_x)
        lut_qy = self.qo.quantize_tensor(lut_y)
        self.register_buffer('lut_qy', lut_qy)
        self.register_buffer('lut_qx', lut_qx)   

    def quantize_inference(self, x):
        y = interp(x, self.lut_qx, self.lut_qy)
        y = y.round_().clamp_(0., 2.**self.num_bits-1.)
        return y

class QMaxPooling2d(QModule):
    def __init__(self, kernel_size=3, stride=1, padding=0, qi=False, num_bits=None):
        super(QMaxPooling2d, self).__init__(qi=qi, num_bits=num_bits)
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = F.max_pool2d(
            input=x, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding
        )
        return x
    
    def quantize_inference(self, x):
        return F.max_pool2d(
            input=x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )

class QAdaptiveAvgPool2d(QModule):
    def __init__(self, qi=False, num_bits=None, pool_module=None):
        super(QAdaptiveAvgPool2d, self).__init__(qi=qi, num_bits=num_bits)
        self.pool_module = pool_module

    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = self.pool_module(x)
        return x
    
    def quantize_inference(self, x):
        return self.pool_module(x).round_()

class QAvgPool2d(QModule):
    def __init__(self, kernel_size=3, stride=1, padding=0, qi=False, num_bits=None):
        super(QAvgPool2d, self).__init__(qi=qi, num_bits=num_bits)
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def freeze(self, qi=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')
        if qi is not None:
            self.qi = qi

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = FakeQuantize.apply(x, self.qi)

        x = F.avg_pool2d(
            input=x, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding
        )
        return x
    
    def quantize_inference(self, x):
        return F.avg_pool2d(
            input=x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        ).round_()

class QLayer(QModule):
    def __init__(self, layer, qi=False, qo=True, num_bits=8):
        super().__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.layer = layer
        self.num_bits = num_bits
        self.qresblock1 = QResBlock(self.layer[0], qi=False, qo=True, num_bits=num_bits)
        self.qresblock2 = QResBlock(self.layer[1], qi=False, qo=True, num_bits=num_bits)

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided')
        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided')
        
        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        self.qresblock1.freeze(qi=self.qi)
        self.qresblock2.freeze(qi=self.qresblock1.qo)

    def forward(self, x):
        x = self.qresblock1(x)
        x = self.qresblock2(x)
        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = FakeQuantize.apply(x, self.qo)

        return x
    
    def quantize_inference(self, x):
        x = self.qresblock1.quantize_inference(x)
        out = self.qresblock2.quantize_inference(x)
        return out
        
class QResBlock(QModule):
    def __init__(self, block, qi=False, qo=True, num_bits=8):
        super().__init__(qi=qi, qo=qo, num_bits=num_bits)
        self.block = block
        self.num_bits = num_bits
        self.conv1 = self.block.left[0]
        self.bn1 = self.block.left[1]        
        self.conv2 = self.block.left[3]
        self.bn2 = self.block.left[4]
        self.qconv_bn_relu_1 = QConvBNReLU(conv_module=self.conv1, bn_module=self.bn1 ,qi=False, qo=True, num_bits=num_bits)
        self.qconv_bn_1 = QConvBN(conv_module=self.conv2, bn_module=self.bn2, qi=False, qo=True, num_bits=num_bits)
        if self.block.shortcut.__len__() != 0:
            print("create a shortcut")
            self.shortcut_conv = self.block.shortcut[0]
            self.shortcut_bn = self.block.shortcut[1]
            self.qshortcut = QConvBN(conv_module=self.shortcut_conv, bn_module=self.shortcut_bn, qi=False, qo=True, num_bits=num_bits)
        self.qadd = QAdd(qo=True, num_bits=num_bits)
        self.qrelu = QRelu()

    def freeze(self, qi=None, qo=None):
        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided')
        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided')
        
        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo

        self.qconv_bn_relu_1.freeze(qi=self.qi)
        self.qconv_bn_1.freeze(qi=self.qconv_bn_relu_1.qo)
        if hasattr(self, 'qshortcut'):
            self.qshortcut.freeze(qi=self.qi)
            self.qadd.freeze(qi_1=self.qconv_bn_1.qo, qi_2=self.qshortcut.qo)
        else:
            self.qadd.freeze(qi_1=self.qconv_bn_1.qo, qi_2=self.qi)

        self.qrelu.freeze(qi=self.qadd.qo)       

    def forward(self, x):
        y1 = self.qconv_bn_relu_1(x)
        y1 = self.qconv_bn_1(y1)
        if hasattr(self, 'qshortcut'):
            print("forward a shortcut")
            y2 = self.qshortcut(x)
            res = self.qadd(y1, y2)
        else:
            res = self.qadd(y1, x)
        res = self.qrelu(res)
        
        if hasattr(self, 'qo'):
            self.qo.update(res)
            res = FakeQuantize.apply(res, self.qo)
            
        return res
    
    def quantize_inference(self, x):
        y1 = self.qconv_bn_relu_1.quantize_inference(x)
        y1 = self.qconv_bn_1.quantize_inference(y1)
        if hasattr(self, 'qshortcut'):
            y2 = self.qshortcut.quantize_inference(x)
            out = self.qadd.quantize_inference(y1, y2)
        else:
            out = self.qadd.quantize_inference(y1, x)

        out = self.qrelu.quantize_inference(out)
        return out
