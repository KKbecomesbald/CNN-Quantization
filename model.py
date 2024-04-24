import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
from quantize import QModule, QConv2d, QLinear, QMaxPooling2d, QAdaptiveAvgPool2d, QAvgPool2d, QRelu, QParam, QConvBNReLU, QAdd, QConvBN, QLayer, QResBlock

class Net(nn.Module):
    def __init__(self, num_channels=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 40, 3, 1)
        self.relu1 = nn.ReLU()
        self.maxpooling1 = nn.MaxPool2d(2, 2, 0)
        self.conv2 = nn.Conv2d(40, 40, 3, 1, groups=20)
        self.relu2 = nn.ReLU()
        self.maxpooling2 = nn.MaxPool2d(2, 2, 0)
        self.fc = nn.Linear(5*5*40, 10)

    def forward(self, x):
        print("x size is", x.size())        
        x = self.conv1(x)
        x = self.relu1(x)
        print("x size is", x.size())
        x = self.maxpooling1(x)
        print("x size is", x.size())
        x = self.conv2(x)
        x = self.relu2(x)
        print("x size is", x.size())
        x = self.maxpooling2(x)
        print("x size is", x.size())
        x = x.view(-1, 5*5*40)
        x = self.fc(x)
        return x

    def quantize(self, num_bits=8):
        print("self.conv1", id(self.conv1))
        self.qconv1 = QConv2d(self.conv1, qi=True, qo=True, num_bits=num_bits)
        print("qconv1", id(self.qconv1.conv_module))
        self.qrelu1 = QRelu()
        self.qmaxpool2d_1 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qconv2 = QConv2d(self.conv2, qi=False, qo=True, num_bits=num_bits)
        self.qrelu2 = QRelu()
        self.qmaxpool2d_2 = QMaxPooling2d(kernel_size=2, stride=2, padding=0)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits)
        self.conv1 = None
        self.relu1 = None
        self.maxpooling1 = None
        self.conv2 = None
        self.relu2 = None
        self.maxpooling2 = None
        self.fc = None

    def quantize_forward(self, x):
        x = self.qconv1(x)
        x = self.qrelu1(x)
        x = self.qmaxpool2d_1(x)
        x = self.qconv2(x)
        x = self.qrelu2(x)
        x = self.qmaxpool2d_2(x)
        x = x.view(-1, 5*5*40)
        x = self.qfc(x)
        return x
    
    def freeze(self):
        self.qconv1.freeze()
        self.qrelu1.freeze(self.qconv1.qo)
        self.qmaxpool2d_1.freeze(self.qconv1.qo)
        self.qconv2.freeze(qi=self.qconv1.qo)
        self.qrelu2.freeze(self.qconv2.qo)
        self.qmaxpool2d_2.freeze(self.qconv2.qo)
        self.qfc.freeze(qi=self.qconv2.qo)

    def quantize_inference(self, x):
        qx = self.qconv1.qi.quantize_tensor(x)
        qx = self.qconv1.quantize_inference(qx)
        qx = self.qrelu1.quantize_inference(qx)
        qx = self.qmaxpool2d_1.quantize_inference(qx)
        qx = self.qconv2.quantize_inference(qx)
        qx = self.qrelu2.quantize_inference(qx)
        qx = self.qmaxpool2d_2.quantize_inference(qx)
        qx = qx.view(-1, 5*5*40)
        qx = self.qfc.quantize_inference(qx)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, ResBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # print("define resnet conv1's weight size is", self.conv1[0].weight.size())
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def quantize(self, num_bits=8):
        self.qconv_bn_relu_1 = QConvBNReLU(self.conv1[0], self.conv1[1], qi=True, qo=True, num_bits=num_bits)
        # print("quantize resnet conv1's weight size is", self.conv1[0].weight.size())
        self.qlayer1 = QLayer(self.layer1, qi=False, qo=True, num_bits=num_bits)
        self.qlayer2 = QLayer(self.layer2, qi=False, qo=True, num_bits=num_bits)
        self.qlayer3 = QLayer(self.layer3, qi=False, qo=True, num_bits=num_bits)
        self.qlayer4 = QLayer(self.layer4, qi=False, qo=True, num_bits=num_bits)
        self.qavgpool = QAdaptiveAvgPool2d(pool_module=self.avgpool)
        self.qfc = QLinear(self.fc, qi=False, qo=True, num_bits=num_bits)
        return
    
    def quantize_forward(self, x):
        # print("resnet x size is", x.size())
        x = self.qconv_bn_relu_1(x)
        # print("resnet conv1's output is", x.size())
        x = self.qlayer1(x)
        # print("resnet layer1's output is", x.size())
        x = self.qlayer2(x)
        # print("resnet layer2's output is", x.size())
        x = self.qlayer3(x)
        # print("resnet layer3's output is", x.size())
        x = self.qlayer4(x)
        # print("resnet layer4's output is", x.size())
        x = self.qavgpool(x)
        x = x.view(x.size(0), -1)
        x = self.qfc(x)
        return x
    
    def freeze(self):
        self.qconv_bn_relu_1.freeze()
        self.qlayer1.freeze(self.qconv_bn_relu_1.qo)
        self.qlayer2.freeze(self.qlayer1.qo)
        self.qlayer3.freeze(self.qlayer2.qo)
        self.qlayer4.freeze(self.qlayer3.qo)
        self.qavgpool.freeze(self.qlayer4.qo)
        self.qfc.freeze(self.qlayer4.qo)

    def quantize_inference(self, x):
        # print("qi scale is", self.qconv_bn_relu_1.qi.scale)
        # print("qi z is", self.qconv_bn_relu_1.qi.zero_point)
        # print("qi min is", self.qconv_bn_relu_1.qi.min)
        # print("qi max is", self.qconv_bn_relu_1.qi.max)
        qx = self.qconv_bn_relu_1.qi.quantize_tensor(x)
        qx = self.qconv_bn_relu_1.quantize_inference(qx)
        qx = self.qlayer1.quantize_inference(qx)
        qx = self.qlayer2.quantize_inference(qx)
        qx = self.qlayer3.quantize_inference(qx)
        qx = self.qlayer4.quantize_inference(qx)
        qx = self.qavgpool.quantize_inference(qx)
        qx = qx.view(qx.size(0), -1)
        qx = self.qfc.quantize_inference(qx)
        out = self.qfc.qo.dequantize_tensor(qx)
        return out

class LeNet(nn.Module):
    def __init__(self, num_channels=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # print("x size is", x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # print("x size is", x.size())
        x = self.pool1(x)
        # print("x size is", x.size())
        x = self.conv2(x)
        x = self.bn2(x)
        # print("x size is", x.size())
        x = F.relu(x)
        x = self.pool2(x)
        # print("x size is", x.size())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
    def quantize(self, num_bits=8):
        self.qconv_bn_relu1 = QConvBNReLU(conv_module=self.conv1, bn_module=self.bn1, qi=True, qo=True, num_bits=num_bits)
        self.qpool1 = QMaxPooling2d(kernel_size=2, stride=2)
        self.qconv_bn_relu2 = QConvBNReLU(conv_module=self.conv2, bn_module=self.bn2, qi=False, qo=True, num_bits=num_bits)
        self.qpool2 = QMaxPooling2d(kernel_size=2, stride=2)
        self.qfc1 = QLinear(fc_module=self.fc1, qi=False, qo=True, num_bits=num_bits)
        self.qrelu3 = QRelu()
        self.qfc2 = QLinear(fc_module=self.fc2, qi=False, qo=True, num_bits=num_bits)
        self.qrelu4 = QRelu()
        self.qfc3 = QLinear(fc_module=self.fc3, qi=False, qo=True, num_bits=num_bits)

    def quantize_forward(self, x):
        x = self.qconv_bn_relu1(x)
        x = self.qpool1(x)
        x = self.qconv_bn_relu2(x)
        x = self.qpool2(x)        
        x = x.view(x.size(0), -1)
        x = self.qfc1(x)
        x = self.qrelu3(x)
        x = self.qfc2(x)
        x = self.qrelu4(x)
        x = self.qfc3(x)
        return x
    
    def freeze(self):
        self.qconv_bn_relu1.freeze()
        self.qpool1.freeze(self.qconv_bn_relu1.qo)
        self.qconv_bn_relu2.freeze(qi=self.qconv_bn_relu1.qo)
        self.qpool2.freeze(self.qconv_bn_relu2.qo)
        self.qfc1.freeze(qi=self.qconv_bn_relu2.qo)
        self.qrelu3.freeze(self.qfc1.qo)
        self.qfc2.freeze(qi=self.qfc1.qo)
        self.qrelu4.freeze(self.qfc2.qo)        
        self.qfc3.freeze(qi=self.qfc2.qo)
    
    def quantize_inference(self, x):
        qx = self.qconv_bn_relu1.qi.quantize_tensor(x)
        qx = self.qconv_bn_relu1.quantize_inference(qx)
        qx = self.qpool1.quantize_inference(qx)
        qx = self.qconv_bn_relu2.quantize_inference(qx)
        qx = self.qpool2.quantize_inference(qx)
        qx = qx.view(x.size(0), -1)
        qx = self.qfc1.quantize_inference(qx)
        qx = self.qrelu3.quantize_inference(qx)
        qx = self.qfc2.quantize_inference(qx)
        qx = self.qrelu4.quantize_inference(qx)
        qx = self.qfc3.quantize_inference(qx)        
        out = self.qfc3.qo.dequantize_tensor(qx)
        return out
    
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            #1 
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),     
            nn.MaxPool2d(kernel_size=2, stride=2),         
            #5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),            
            #7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=2, stride=2), 
            #8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            #11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1, stride=1),            
            )
        self.classifier = nn.Sequential(
            #14
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            #15
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            #16
            nn.Linear(4096, num_classes)
            )
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def quantize(self, num_bits=8):
        self.qconv_bn_relu1 = QConvBNReLU(
            conv_module=self.features[0],
            bn_module=self.features[1],
            qi=True, qo=True, num_bits=num_bits)
        
        self.qconv_bn_relu2 = QConvBNReLU(
            conv_module=self.features[3],
            bn_module=self.features[4],
            qi=False, qo=True, num_bits=num_bits)
        
        self.qpool1 = QMaxPooling2d(kernel_size=2, stride=2)

        self.qconv_bn_relu3 = QConvBNReLU(
            conv_module=self.features[7],
            bn_module=self.features[8],
            qi=False, qo=True, num_bits=num_bits)
        
        self.qconv_bn_relu4 = QConvBNReLU(
            conv_module=self.features[10],
            bn_module=self.features[11],
            qi=False, qo=True, num_bits=num_bits)
        
        self.qpool2 = QMaxPooling2d(kernel_size=2, stride=2)

        self.qconv_bn_relu5 = QConvBNReLU(
            conv_module=self.features[14],
            bn_module=self.features[15],
            qi=False, qo=True, num_bits=num_bits)
        
        self.qconv_bn_relu6 = QConvBNReLU(
            conv_module=self.features[17],
            bn_module=self.features[18],
            qi=False, qo=True, num_bits=num_bits)

        self.qconv_bn_relu7 = QConvBNReLU(
            conv_module=self.features[20],
            bn_module=self.features[21],
            qi=False, qo=True, num_bits=num_bits)

        self.qpool3 = QMaxPooling2d(kernel_size=2, stride=2)

        self.qconv_bn_relu8 = QConvBNReLU(
            conv_module=self.features[24],
            bn_module=self.features[25],
            qi=False, qo=True, num_bits=num_bits)

        self.qconv_bn_relu9 = QConvBNReLU(
            conv_module=self.features[27],
            bn_module=self.features[28],
            qi=False, qo=True, num_bits=num_bits)

        self.qconv_bn_relu10 = QConvBNReLU(
            conv_module=self.features[30],
            bn_module=self.features[31],
            qi=False, qo=True, num_bits=num_bits)
        
        self.qpool4 = QMaxPooling2d(kernel_size=2, stride=2)

        self.qconv_bn_relu11 = QConvBNReLU(
            conv_module=self.features[34],
            bn_module=self.features[35],
            qi=False, qo=True, num_bits=num_bits)
        
        self.qconv_bn_relu12 = QConvBNReLU(
            conv_module=self.features[37],
            bn_module=self.features[38],
            qi=False, qo=True, num_bits=num_bits)

        self.qconv_bn_relu13 = QConvBNReLU(
            conv_module=self.features[40],
            bn_module=self.features[41],
            qi=False, qo=True, num_bits=num_bits)

        self.qpool5 = QMaxPooling2d(kernel_size=2, stride=2)
        self.qpool6 = QAvgPool2d(kernel_size=1, stride=1)

        self.qfc1 = QLinear(
            fc_module=self.classifier[0], 
            qi=False, qo=True,
            num_bits=num_bits)
        
        self.qrelu1 = QRelu()

        self.qfc2 = QLinear(
            fc_module=self.classifier[3], 
            qi=False, qo=True,
            num_bits=num_bits)

        self.qrelu2 = QRelu()

        self.qfc3 = QLinear(
            fc_module=self.classifier[6], 
            qi=False, qo=True,
            num_bits=num_bits)
        
    def quantize_forward(self, x):
        x = self.qconv_bn_relu1(x)
        x = self.qconv_bn_relu2(x)
        x = self.qpool1(x)
        x = self.qconv_bn_relu3(x)
        x = self.qconv_bn_relu4(x)
        x = self.qpool2(x)
        x = self.qconv_bn_relu5(x)
        x = self.qconv_bn_relu6(x)
        x = self.qconv_bn_relu7(x)
        x = self.qpool3(x)
        x = self.qconv_bn_relu8(x)
        x = self.qconv_bn_relu9(x)
        x = self.qconv_bn_relu10(x)
        x = self.qpool4(x)
        x = self.qconv_bn_relu11(x)
        x = self.qconv_bn_relu12(x)
        x = self.qconv_bn_relu13(x)
        x = self.qpool5(x)
        x = self.qpool6(x)
        x = x.view(x.size(0), -1)
        x = self.qfc1(x)
        x = self.qrelu1(x)
        x = self.qfc2(x)
        x = self.qrelu2(x)
        x = self.qfc3(x)
        return x

    def freeze(self):
        self.qconv_bn_relu1.freeze()
        self.qconv_bn_relu2.freeze(qi=self.qconv_bn_relu1.qo)
        self.qpool1.freeze(qi=self.qconv_bn_relu2.qo)
        self.qconv_bn_relu3.freeze(qi=self.qconv_bn_relu2.qo)
        self.qconv_bn_relu4.freeze(qi=self.qconv_bn_relu3.qo)
        self.qpool2.freeze(qi=self.qconv_bn_relu4.qo)
        self.qconv_bn_relu5.freeze(qi=self.qconv_bn_relu4.qo)
        self.qconv_bn_relu6.freeze(qi=self.qconv_bn_relu5.qo)
        self.qconv_bn_relu7.freeze(qi=self.qconv_bn_relu6.qo)
        self.qpool3.freeze(qi=self.qconv_bn_relu7.qo)        
        self.qconv_bn_relu8.freeze(qi=self.qconv_bn_relu7.qo)
        self.qconv_bn_relu9.freeze(qi=self.qconv_bn_relu8.qo)
        self.qconv_bn_relu10.freeze(qi=self.qconv_bn_relu9.qo)
        self.qpool4.freeze(qi=self.qconv_bn_relu10.qo)    
        self.qconv_bn_relu11.freeze(qi=self.qconv_bn_relu10.qo)
        self.qconv_bn_relu12.freeze(qi=self.qconv_bn_relu11.qo)
        self.qconv_bn_relu13.freeze(qi=self.qconv_bn_relu12.qo)
        self.qpool5.freeze(qi=self.qconv_bn_relu12.qo)    
        self.qpool6.freeze(qi=self.qconv_bn_relu12.qo)    
        self.qfc1.freeze(qi=self.qconv_bn_relu13.qo)
        self.qrelu1.freeze(qi=self.qfc1.qo)
        self.qfc2.freeze(qi=self.qfc1.qo)
        self.qrelu2.freeze(qi=self.qfc2.qo)
        self.qfc3.freeze(qi=self.qfc2.qo)

    def quantize_inference(self, x):
        qx = self.qconv_bn_relu1.qi.quantize_tensor(x)
        qx = self.qconv_bn_relu1.quantize_inference(qx)
        qx = self.qconv_bn_relu2.quantize_inference(qx)
        qx = self.qpool1.quantize_inference(qx)
        qx = self.qconv_bn_relu3.quantize_inference(qx)
        qx = self.qconv_bn_relu4.quantize_inference(qx)
        qx = self.qpool2.quantize_inference(qx)
        qx = self.qconv_bn_relu5.quantize_inference(qx)
        qx = self.qconv_bn_relu6.quantize_inference(qx)
        qx = self.qconv_bn_relu7.quantize_inference(qx)
        qx = self.qpool3.quantize_inference(qx)
        qx = self.qconv_bn_relu8.quantize_inference(qx)
        qx = self.qconv_bn_relu9.quantize_inference(qx)
        qx = self.qconv_bn_relu10.quantize_inference(qx)
        qx = self.qpool4.quantize_inference(qx)
        qx = self.qconv_bn_relu11.quantize_inference(qx)
        qx = self.qconv_bn_relu12.quantize_inference(qx)
        qx = self.qconv_bn_relu13.quantize_inference(qx)
        qx = self.qpool5.quantize_inference(qx)
        qx = self.qpool6.quantize_inference(qx)
        qx = qx.view(qx.size(0), -1)
        qx = self.qfc1.quantize_inference(qx)
        qx = self.qrelu1.quantize_inference(qx)
        qx = self.qfc2.quantize_inference(qx)
        qx = self.qrelu2.quantize_inference(qx)
        qx = self.qfc3.quantize_inference(qx)
        out = self.qfc3.qo.dequantize_tensor(qx)
        return out