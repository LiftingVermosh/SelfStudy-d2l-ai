# 这个文件是从0实现的多层感知器，并用Fashion-MNIST数据集进行训练和测试。
# 代码主要参考了PyTorch官方文档，并在其基础上进行了修改。

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def load_data_from_fashion_mnist():
    """加载Fashion-MNIST数据集"""
    train_data = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = torchvision.datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    return train_data, test_data

class MLP(nn.Module):
    """手动实现的多层感知器"""
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super().__init__()
        # 初始化参数
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
        
    def forward(self, X):
        X = X.reshape(-1, num_inputs)  # 展平图像
        H = torch.relu(X @ self.W1 + self.b1)  
        return H @ self.W2 + self.b2

def evaluate_accuracy(net, data_loader):
    """计算模型在数据集上的准确率"""
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            y_hat = net(X)
            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

if __name__ == '__main__':
    # 超参数设置
    batch_size = 256
    num_inputs = 28 * 28
    num_outputs = 10
    num_hiddens = 256
    epochs = 10
    lr = 0.01
    
    # 加载数据
    train_data, test_data = load_data_from_fashion_mnist()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    net = MLP(num_inputs, num_hiddens, num_outputs)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    # 训练过程记录
    train_losses = []
    train_accs = []
    test_accs = []
    
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^9} | {'Test Acc':^9}")
    print("-" * 50)
    
    # 训练循环
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss_fn(y_hat, y)
            l.backward()
            optimizer.step()
            epoch_loss += l.item() * X.size(0)
        
        # 计算本epoch平均训练损失
        epoch_loss /= len(train_data)
        train_losses.append(epoch_loss)
        
        # 计算训练集和测试集准确率
        train_acc = evaluate_accuracy(net, train_loader)
        test_acc = evaluate_accuracy(net, test_loader)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # 输出训练进度
        print(f"{epoch+1:^7} | {epoch_loss:^12.4f} | {train_acc:^9.4f} | {test_acc:^9.4f}")
    
    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'Multilayer Perceptron\\Output Figures\\training_metrics(f0)-{num_hiddens}-{lr}-{epochs}.png')
    plt.show()
    
    # 最终测试准确率
    final_test_acc = evaluate_accuracy(net, test_loader)
    print(f"\nFinal Test Accuracy: {final_test_acc:.4f}")
