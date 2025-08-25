import torch 
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_data_from_fashion_mnist():
    """加载Fashion-MNIST数据集"""
    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor())
    return train_dataset, test_dataset

def dropout_layer(X, dropout):
    """Dropout层"""
    if dropout == 0.0:
        return X  # 没有dropout，返回原输入
    if dropout == 1.0:
        return torch.zeros_like(X)  # 全部丢弃，返回零
    # mask: 0-1的随机张量，dropout概率下部分神经元被丢弃
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)  # 缩放以保持期望值

class net(nn.Module):
    """定义网络结构"""
    def __init__(self, num_inputs, num_outputs, num_hiddens, is_training=True):
        super(net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training  # 使用自定义训练标志
        self.lin1 = nn.Linear(num_inputs, num_hiddens[0])
        self.lin2 = nn.Linear(num_hiddens[0], num_hiddens[1])
        self.lin3 = nn.Linear(num_hiddens[1], num_outputs)
        self.relu = nn.ReLU()
    
    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        
        if self.training:  # 只在训练时应用dropout
            H1 = dropout_layer(H1, 0.2)
        
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, 0.5)
        
        out = self.lin3(H2)
        return out

def evaluate(net, test_loader):
    """评估模型在测试集上的准确率"""
    net.eval()  # 设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape((-1, 784))
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def train(net, train_dataset, test_dataset, num_epochs, batch_size=64):
    """训练网络并记录指标"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # 记录训练过程
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        net.train()  # 设置为训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape((-1, 784))
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'
                     .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
        
        # 计算epoch平均损失和训练准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_train_acc)
        
        # 计算测试准确率
        epoch_test_acc = evaluate(net, test_loader)
        test_accuracies.append(epoch_test_acc)
        
        print('Epoch [{}/{}], Average Loss: {:.4f}, Train Acc: {:.2f}%, Test Acc: {:.2f}%'
              .format(epoch+1, num_epochs, epoch_loss, epoch_train_acc, epoch_test_acc))
    
    # 可视化结果
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies, 'r-', label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), test_accuracies, 'g-', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('Dropout_training_results.png')  # 保存图像
    plt.show()
    
    return train_losses, train_accuracies, test_accuracies

if __name__ == '__main__':
    # 加载数据集
    train_dataset, test_dataset = load_data_from_fashion_mnist()
    # 定义网络结构
    net = net(num_inputs=784, num_outputs=10, num_hiddens=[500, 200], is_training=True)
    # 训练网络并可视化结果
    train_losses, train_accs, test_accs = train(net, train_dataset, test_dataset, num_epochs=10)
