import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.training_visualizer import TrainingVisualizer  

class _DenseLayer(nn.Module):
    """ 单个 Dense 层 """
    def __init__(self, num_input_features, growth_rate):
        super().__init__()
        # 瓶颈层结构：BN - ReLU - 1x1 Conv
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, pre_features):
        cat_features = torch.cat(pre_features, 1)

        out = self.conv1(F.relu(self.bn1(cat_features)))
        out = self.conv2(F.relu(self.bn2(out)))

        return out
    
class DenseBlock(nn.Module):
    """ 完整的 Dense 块 """
    def __init__(self, num_layers, num_input_features, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i*growth_rate,     # 输入通道随层数增加
                growth_rate)
            self.layers.append(layer)

    def forward(self, init_features):
        features = [init_features]
        for layer in self.layers:
            new_features = layer(features)       # 逐层更新特征图
            features.append(new_features)
        return torch.cat(features, 1)       # 拼接特征图

class TransitionLayer(nn.Module):
    """ 过渡层 """
    def __init__(self, num_input_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_input_features//2, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)    # 平均池化

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out

class DenseNet(nn.Module):
    """ DenseNet 网络 """
    def __init__(self, num_init_features=24, growth_rate=12, block_config=[16, 16, 16], num_classes=10):
        super().__init__()
        # 此处修改了初始层：原论文卷积核尺寸过大，此处使用kernel_size=3, stride=1, padding=1，移除MaxPool，适应CIFAR-10的32x32输入
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True)
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers, num_input_features=num_features, growth_rate=growth_rate)
            self.features.add_module('denseblock%d' % (i+1), block)
            num_features = num_features + num_layers * growth_rate
            # 在最后一个块之前添加过渡层
            if i != len(block_config)-1:
                trans = TransitionLayer(num_input_features=num_features)
                self.features.add_module('transition%d' % (i+1), trans)
                num_features = num_features // 2

        # 全局平均池化和分类器
        self.final_bn = nn.BatchNorm2d(num_features)  # 最终BatchNorm
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(self.final_bn(features), inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def train(model, device, train_loader, optimizer, epoch, visualizer=None):
    """ 训练函数 """
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
        
        total_loss += loss.item() * data.size(0)
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
    """ 测试函数 """
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
    
    # 实例化DenseNet，使用CIFAR-10优化配置
    model = DenseNet(
        num_init_features=24,  # 初始通道数
        growth_rate=12,        # 增长率
        block_config=[16, 16, 16],  # 每个DenseBlock的层数，对应DenseNet-BC-100
        num_classes=10         # CIFAR-10有10类
    ).to(device)
    
    # 优化器：使用SGD with momentum，学习率0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    
    # 数据加载：添加数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    
    train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10('data', train=False, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 学习率调度器：每50轮减少学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    # 创建TrainingVisualizer实例
    visualizer = TrainingVisualizer()
    
    # 训练循环
    for epoch in range(1, 21):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, visualizer)
        test_loss, test_acc = test(model, device, test_loader, visualizer, epoch)
        scheduler.step()  # 更新学习率
    
    # 保存模型和绘制曲线
    target_path = 'Models_Output'
    os.makedirs(target_path, exist_ok=True)
    visualizer.plot(os.path.join(target_path, 'DenseNet-CIFAR10.png'))
