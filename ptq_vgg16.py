from torch.serialization import load
from model import VGG16
import onnx
import onnx.utils
import torch 
from torchsummary import summary
from torchvision import datasets, transforms
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os.path as osp
import os

def direct_quantize(model, test_loader):
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_forward(data)
        if i % 20 == 0:
            break
    print('direct quantization finish')

def full_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Full Model Accuracy: {:.2f}%\n'.format(100. * correct / len(test_loader.dataset)))

def quantize_inference(model, test_loader):
    correct = 0
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_inference(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nTest set: Quant Model Accuracy: {:.2f}%\n'.format(100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 128
    # load_quant_model_file = 'ckpt/CIFAR10_resnet18_ptq.pth'
    load_quant_model_file = None
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    model = VGG16(num_classes=10).to(device)
    model.load_state_dict(torch.load('ckpt/CIFAR10_vgg16.pth'))
    
    #eport onnx model of resnet
    x = torch.randn(BATCH_SIZE, 3, 32, 32, requires_grad=True)
    torch.onnx.export(model,
                      x,
                      "ckpt/vgg16.onnx",
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output']
                      )
    model_file = 'ckpt/vgg16.onnx'
    onnx_model = onnx.load(model_file)
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), model_file)

    save_file = 'ckpt/CIFAR10_vgg16_ptq.pth'

    model.eval()
    full_inference(model, testloader)
    # summary(model, (1, 28, 28), device='cuda')
    num_bits = 2
    model.quantize(num_bits=num_bits)
    # summary(model, (1, 28, 28), device='cuda')
    # print("M size of model is",model.qconv_bn_relu_1.M.size())
    # print("M of model is",model.qconv_bn_relu_1.M.data)
    model.eval()
    print('Quantization bit: %d' % num_bits)

    if load_quant_model_file is not None:
        model.load_state_dict(torch.load(load_quant_model_file))
        print(model.state_dict)
        print('Successfully load quantized model %s' % load_quant_model_file)
    else:
        direct_quantize(model, trainloader)
        model.freeze()
        # print(model.state_dict)     
        torch.save(model.state_dict(), save_file)
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor])
    quantize_inference(model, testloader)