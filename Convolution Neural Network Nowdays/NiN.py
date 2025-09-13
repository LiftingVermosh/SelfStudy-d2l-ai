import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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


def train(model, device, train_loader, optimizer, criterion, epoch):
    """ 训练函数 """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}"
                  f" ({100. * batch_idx/len(train_loader):.0f}%)]\tLoss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader.dataset)
    acc = 100. * correct / total
    print(f"Train Epoch {epoch} Summary: Loss={avg_loss:.4f} Acc={acc:.2f}%")

def test(model, device, test_loader, criterion):
    """ 测试函数 """
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)\n")
    return acc


if __name__ == '__main__':
    # 数据增强+归一化
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914,0.4822,0.4465],
                             std=[0.2023,0.1994,0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914,0.4822,0.4465],
                             std=[0.2023,0.1994,0.2010])
    ])

    # 数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型 & 损失 & 优化器 & 学习率调度
    model = NiN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # 训练
    epochs = 150
    best_acc = 0
    for epoch in range(1, epochs+1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        acc = test(model, device, test_loader, criterion)
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "nin_best.pth")
            print(f"Best model saved at epoch {epoch} with Acc={acc:.2f}%")

    print(f"Training finished. Best Test Acc: {best_acc:.2f}%")
