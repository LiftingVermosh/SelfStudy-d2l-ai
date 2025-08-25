# 这个文件是从0实现的多层感知器，并用Fashion-MNIST数据集进行训练和测试。
# 此代码支持GPU加速训练

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 检查GPU可用性并选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        self.num_inputs = num_inputs
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens[0], device=device) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens[0], device=device))
        self.W2 = nn.Parameter(torch.randn(num_hiddens[0], num_hiddens[1], device=device) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(num_hiddens[1], device=device))
        self.W3 = nn.Parameter(torch.randn(num_hiddens[1], num_outputs, device=device) * 0.01)
        self.b3 = nn.Parameter(torch.zeros(num_outputs, device=device))
        
    def forward(self, X):
        # 确保输入数据在正确设备上
        X = X.to(device)
        X = X.reshape(-1, self.num_inputs)  # 展平图像
        H1 = torch.relu(X @ self.W1 + self.b1)
        H2 = torch.relu(H1 @ self.W2 + self.b2)
        return H2 @ self.W3 + self.b3

def evaluate_accuracy(net, data_loader):
    """计算模型在数据集上的准确率"""
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            # 将数据移动到与模型相同的设备
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

if __name__ == '__main__':
    # 超参数设置
    batch_size = 512    # 增大批大小以充分利用GPU并行能力
    num_inputs = 28 * 28
    num_outputs = 10
    num_hiddens = [512, 256]   # GPU可以处理更大模型
    epochs = 20         # 增加epoch数，GPU训练更快
    lr = 0.05
    
    # 加载数据
    train_data, test_data = load_data_from_fashion_mnist()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # 初始化模型、损失函数和优化器
    net = MLP(num_inputs, num_hiddens, num_outputs)
    net.to(device)  # 将模型移动到GPU
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)  # 添加动量加速收敛
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    
    # 训练过程记录
    train_losses = []
    train_accs = []
    test_accs = []
    
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^9} | {'Test Acc':^9} | {'LR':^8}")
    print("-" * 60)
    
    # 训练循环
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        for X, y in train_loader:
            # 将数据移动到GPU
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
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
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(test_acc)  # 根据测试准确率调整学习率
        
        # 输出训练进度
        print(f"{epoch+1:^7} | {epoch_loss:^12.4f} | {train_acc:^9.4f} | {test_acc:^9.4f} | {current_lr:^8.5f}")
    
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
    plt.savefig(f'Multilayer Perceptron\\Output Figures\\GPU_training_metrics-{num_hiddens}-{lr}-{epochs}-{len(num_hiddens)}.png')
    plt.show()
    
    # 最终测试准确率
    final_test_acc = evaluate_accuracy(net, test_loader)
    print(f"\nFinal Test Accuracy: {final_test_acc:.4f}")
    
    # 保存模型
    # torch.save(net.state_dict(), f'Multilayer Perceptron\Output Figures\\mlp_gpu_model-{final_test_acc:.4f}.pth')
    # print("Model saved to disk.")
