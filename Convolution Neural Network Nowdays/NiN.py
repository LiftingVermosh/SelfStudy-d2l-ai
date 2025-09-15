import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.training_visualizer import TrainingVisualizer

class NiNBlock(nn.Module):
    """ NiN 块 """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class NiN(nn.Module):
    """ NiN 网络 """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            NiNBlock(3, 192, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(p=0.3),

            NiNBlock(192, 192, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(p=0.4),

            NiNBlock(192, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout2d(p=0.5),

            NiNBlock(256, num_classes, kernel_size=3, stride=1, padding=1)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train(model, device, train_loader, optimizer, epoch, visualizer=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)  # 累加损失
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    avg_loss = total_loss / total
    acc = 100. * correct / total
    if visualizer is not None:
        visualizer.update_train(epoch, avg_loss, acc)
    print(f'Train Epoch {epoch}: Average Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%')
    return avg_loss, acc

def test(model, device, test_loader, visualizer=None, epoch=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    if visualizer is not None and epoch is not None:
        visualizer.update_test(epoch, test_loss, acc)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return test_loss, acc


if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NiN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 加载 CIFAR-10 数据集
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

    # 创建 TrainingVisualizer 实例
    visualizer = TrainingVisualizer()
    
    # 训练循环
    for epoch in range(1, 21):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, visualizer)
        test_loss, test_acc = test(model, device, test_loader, visualizer, epoch)
    
    # 绘制训练曲线
    target_path = 'Models_Output'
    os.makedirs(target_path, exist_ok=True)
    visualizer.plot(os.path.join(target_path, 'NiN-CIFAR10.png'))
