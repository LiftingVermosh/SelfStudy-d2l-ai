import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Inception(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 分支1: 1x1卷积
        self.module_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], kernel_size=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True)
        )
        # 分支2: 1x1卷积 -> 3x3卷积
        self.module_2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),  # 减少维度
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels[1]),
            nn.ReLU(inplace=True)
        )
        # 分支3: 1x1卷积 -> 5x5卷积
        self.module_3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels[2], kernel_size=5, padding=2),  # padding=2保持尺寸
            nn.BatchNorm2d(out_channels[2]),
            nn.ReLU(inplace=True)
        )
        # 分支4: 3x3最大池化 -> 1x1卷积
        self.module_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # stride=1保持尺寸
            nn.Conv2d(in_channels, out_channels[3], kernel_size=1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x1 = self.module_1(x)
        x2 = self.module_2(x)
        x3 = self.module_3(x)
        x4 = self.module_4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 预处理层：输出尺寸16x16x192
        self.preLayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        # 下采样层：将尺寸从16x16下采样到8x8
        self.maxPoolingLayer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Inception模块序列：所有模块保持8x8尺寸
        self.inception_1 = nn.Sequential(   
            Inception(192, [128, 64, 32, 64]),  # 输出通道288
            Inception(288, [128, 64, 32, 64]),  # 输出通道288
        )
        self.inception_2 = nn.Sequential(
            Inception(288, [128, 96, 48, 64]),  # 输出通道336
            Inception(336, [128, 96, 48, 64]),  # 输出通道336
            Inception(336, [128, 96, 48, 64]),  # 输出通道336
            Inception(336, [128, 96, 48, 64]),  # 输出通道336
            Inception(336, [128, 96, 48, 64]),  # 输出通道336
        )
        self.inception_3 = nn.Sequential(
            Inception(336, [192, 96, 48, 64]),  # 输出通道400
            Inception(400, [192, 96, 48, 64]),   # 输出通道400
        )
        # 全局平均池化和全连接层
        self.avgPoolingLayer = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(400, 10)  # 输入特征数为400，输出10类（CIFAR-10）
    
    def forward(self, x):
        x = self.preLayer(x) 
        x = self.maxPoolingLayer(x) 
        x = self.inception_1(x) 
        x = self.inception_2(x)  
        x = self.inception_3(x) 
        x = self.avgPoolingLayer(x)  
        x = x.view(x.size(0), -1) 
        x = self.fc(x) 
        return x

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # 将一批的损失相加
            pred = output.argmax(dim=1, keepdim=True)  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))) 

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoogLeNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_loader = DataLoader(datasets.CIFAR10('data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914,0.4822,0.4465],
                             std=[0.2023,0.1994,0.2010])
    ])), batch_size=128, shuffle=True)
    test_loader = DataLoader(datasets.CIFAR10('data', train=False, transform=transforms.Compose([    
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914,0.4822,0.4465],
                             std=[0.2023,0.1994,0.2010])
    ])), batch_size=128, shuffle=False)
    for epoch in range(1, 21):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)