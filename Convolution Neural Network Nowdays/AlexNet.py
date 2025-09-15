import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.training_visualizer import TrainingVisualizer

class AlexNet(nn.Module):
    """ AlexNet 类 """
    def __init__(self, num_classes=1000):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 自适应池化层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),   
            nn.Flatten(),
            nn.Linear(256, 4096), nn.ReLU(inplace=True),
            nn.Dropout(0.5),        # 减轻过拟合
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Dropout(0.5),        # 减轻过拟合
            nn.Linear(4096, num_classes)
        )

    def forward(self,X):
        X = self.squeeze(X)
        X = self.classifier(X)
        return X
    
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
    model = AlexNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

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
    visualizer.plot(os.path.join(target_path, 'AlexNet-CIFAR10.png'))