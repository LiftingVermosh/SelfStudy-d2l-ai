import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16*7*7, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )
    
    def forward(self, x):
        return self.net(x)

# class LeNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
#         self.sigmoid1 = nn.Sigmoid()
#         self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
#         self.sigmoid2 = nn.Sigmoid()
#         self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
    
#     def forward(self, x):
#         x = self.sigmoid1(self.conv1(x))
#         x = self.avgpool1(x)
        
#         x = self.sigmoid2(self.conv2(x))
#         x = self.avgpool2(x)
        
#         x = torch.flatten(x, 1)  # 展平所有维度除了batch
#         x = F.sigmoid(self.fc1(x))
#         x = F.sigmoid(self.fc2(x))
#         x = self.fc3(x)
#         return x


def train(model, device, train_loader, optimizer, epoch):
    # 设置模型为训练模式
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    # 训练模型
    for batch_icx,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        output = model(data)
        
        # 计算损失
        loss = F.cross_entropy(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_icx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_icx * len(data), len(train_loader.dataset),
                100. * batch_icx / len(train_loader), loss.item()))

    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss / len(train_loader), correct, total,
        100. * correct / total))

if __name__ == '__main__':
    # 检查是否有可用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 定义数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # 定义模型
    model = LeNet().to(device)

    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

    # 训练模型
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, epoch)

    # 保存模型
    torch.save(model.state_dict(), 'lenet.pth')
    print('Model saved to lenet.pth')