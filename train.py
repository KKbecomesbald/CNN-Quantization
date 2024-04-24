from model import Net, LeNet, ResNet, ResBlock
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os.path as osp
import os

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    lossLayer = nn.CrossEntropyLoss(reduction='sum')
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossLayer(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{}]\tloss: {:.6f}'.format(
                epoch, 
                batch_idx * len(data), 
                len(train_loader.dataset), 
                loss.item()
                )
            )

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    lossLayer = nn.CrossEntropyLoss(reduction='sum')
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset)
    ))

if __name__=='__main__':
    batch_size = 64
    test_batch_size = 64
    seed = 1
    epochs = 15
    lr = 0.005
    momentum = 0.5
    save_model = True
    type = 2
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(
        datasets.MNIST(
            'data', 
            train=True, 
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    test_loader = DataLoader(
        datasets.MNIST(
            'data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    if type == 1:
        model = Net().to(device)
    elif type == 2:
        model = LeNet(num_channels=1).to(device)
    else:
        model = ResNet(ResBlock).to(device)
        
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    if save_model:
        if not osp.exists('ckpt'):
            os.makedirs('ckpt')
        
        if type == 1:
            torch.save(model.state_dict(), 'ckpt/mnist_cnn.pth')
        elif type == 2:
            torch.save(model.state_dict(), 'ckpt/mnist_lenet.pth')
        else:
            torch.save(model.state_dict(), 'ckpt/mnist_resnet18.pth')