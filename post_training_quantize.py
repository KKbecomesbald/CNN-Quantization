from torch.serialization import load
from model import Net, LeNet
import argparse
import torch 
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp

def direct_quantize(model, test_loader):
    for i, (data, target) in enumerate(test_loader, 1):
        output = model.quantize_forward(data)
        if i % 500 == 0:
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
    batch_size = 64
    load_quant_model_file = None
    type = 2

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, 
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
    )

    if type == 1:
        model = Net()
        model.load_state_dict(torch.load('ckpt/mnist_cnn.pth'))
        save_file = 'ckpt/mnist_cnn_ptq.pth'
    else:
        model = LeNet(num_channels=1)
        model.load_state_dict(torch.load('ckpt/mnist_lenet.pth'))
        save_file = 'ckpt/mnist_lenet_ptq.pth'

    model.eval()
    full_inference(model, test_loader)
    # summary(model, (1, 28, 28), device='cuda')
    num_bits = 32
    model.quantize(num_bits=num_bits)
    # summary(model, (1, 28, 28), device='cuda')
    model.eval()
    print('Quantization bit: %d' % num_bits)

    if load_quant_model_file is not None:
        model.load_state_dict(torch.load(load_quant_model_file))
        print('Successfully load quantized model %s' % load_quant_model_file)

    direct_quantize(model, train_loader)
    model.freeze()
    torch.save(model.state_dict(), save_file)

    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor])

    quantize_inference(model, test_loader)