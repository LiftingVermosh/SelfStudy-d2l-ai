import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from collections import defaultdict

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 定义数据变换和数据加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据集
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义网络结构，添加激活值捕获功能
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rates=[0.2, 0.5], use_gaussian_noise=False, noise_std=0.1):
        super(NeuralNet, self).__init__()
        self.dropout_rates = dropout_rates
        self.use_gaussian_noise = use_gaussian_noise
        self.noise_std = noise_std
        self.activations = {}  # 存储激活值
        self.hooks = []  # 存储钩子
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(hidden_sizes[1], output_size)
        )
        
        # 注册前向钩子来捕获激活值
        self._register_hooks()
    
    def _register_hooks(self):
        """注册钩子来捕获隐藏层的激活值"""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        # 为每个ReLU层注册钩子
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.ReLU):
                self.hooks.append(layer.register_forward_hook(get_activation(f'relu_{i}')))
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        # 如果使用高斯噪声，在输入层添加噪声
        if self.use_gaussian_noise and self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # 清空之前的激活值
        self.activations = {}
        
        return self.layers(x)
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()

# 计算测试损失
def calculate_test_loss(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    
    return test_loss / len(test_loader)

# 评估模型准确率
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# 训练函数，添加激活值方差记录
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, model_name=""):
    model.train()
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    # 记录每个隐藏层的激活值方差
    activation_variances = defaultdict(list)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 训练阶段
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计信息
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 计算epoch训练指标
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        # 计算epoch测试指标
        epoch_test_loss = calculate_test_loss(model, test_loader, criterion)
        epoch_test_acc = evaluate_model(model, test_loader)
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_acc)
        
        # 计算并记录激活值方差
        model.eval()
        with torch.no_grad():
            # 使用一个批次的数据计算激活值方差
            sample_images, _ = next(iter(test_loader))
            sample_images = sample_images.to(device)
            _ = model(sample_images)
            
            for layer_name, activation in model.activations.items():
                variance = torch.var(activation).item()
                activation_variances[layer_name].append(variance)
        
        model.train()
        
        print(f'{model_name} Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, '
              f'Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%')
    
    return train_losses, train_accuracies, test_losses, test_accuracies, activation_variances

# 比较不同配置的实验
def run_experiments():
    num_epochs = 20
    results = {}
    
    # 不使用Dropout
    print("="*50)
    print("实验1: 不使用Dropout")
    model_no_dropout = NeuralNet(784, [500, 200], 10, dropout_rates=[0, 0]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_no_dropout.parameters(), lr=0.001)
    
    start_time = time.time()
    train_losses, train_accuracies, test_losses, test_accuracies, activation_variances = train_model(
        model_no_dropout, train_loader, test_loader, criterion, optimizer, num_epochs, "无Dropout"
    )
    training_time = time.time() - start_time
    
    results['no_dropout'] = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'activation_variances': activation_variances,
        'training_time': training_time
    }
    
    # 使用Dropout [0.2, 0.5]
    print("="*50)
    print("实验2: 使用Dropout [0.2, 0.5]")
    model_dropout = NeuralNet(784, [500, 200], 10, dropout_rates=[0.2, 0.5]).to(device)
    optimizer = optim.Adam(model_dropout.parameters(), lr=0.001)
    
    start_time = time.time()
    train_losses, train_accuracies, test_losses, test_accuracies, activation_variances = train_model(
        model_dropout, train_loader, test_loader, criterion, optimizer, num_epochs, "有Dropout"
    )
    training_time = time.time() - start_time
    
    results['dropout'] = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'activation_variances': activation_variances,
        'training_time': training_time
    }
    
    # 使用权重衰减 (L2正则化)
    print("="*50)
    print("实验3: 使用权重衰减 (L2正则化)")
    model_wd = NeuralNet(784, [500, 200], 10, dropout_rates=[0, 0]).to(device)
    optimizer = optim.Adam(model_wd.parameters(), lr=0.001, weight_decay=1e-4)
    
    start_time = time.time()
    train_losses, train_accuracies, test_losses, test_accuracies, activation_variances = train_model(
        model_wd, train_loader, test_loader, criterion, optimizer, num_epochs, "权重衰减"
    )
    training_time = time.time() - start_time
    
    results['weight_decay'] = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'activation_variances': activation_variances,
        'training_time': training_time
    }
    
    # 同时使用Dropout和权重衰减
    print("="*50)
    print("实验4: 同时使用Dropout和权重衰减")
    model_both = NeuralNet(784, [500, 200], 10, dropout_rates=[0.2, 0.5]).to(device)
    optimizer = optim.Adam(model_both.parameters(), lr=0.001, weight_decay=1e-4)
    
    start_time = time.time()
    train_losses, train_accuracies, test_losses, test_accuracies, activation_variances = train_model(
        model_both, train_loader, test_loader, criterion, optimizer, num_epochs, "Dropout+权重衰减"
    )
    training_time = time.time() - start_time
    
    results['both'] = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'activation_variances': activation_variances,
        'training_time': training_time
    }
    
    # 使用高斯噪声替代Dropout
    print("="*50)
    print("实验5: 使用高斯噪声替代Dropout")
    model_gaussian = NeuralNet(784, [500, 200], 10, dropout_rates=[0, 0], 
                              use_gaussian_noise=True, noise_std=0.1).to(device)
    optimizer = optim.Adam(model_gaussian.parameters(), lr=0.001)
    
    start_time = time.time()
    train_losses, train_accuracies, test_losses, test_accuracies, activation_variances = train_model(
        model_gaussian, train_loader, test_loader, criterion, optimizer, num_epochs, "高斯噪声"
    )
    training_time = time.time() - start_time
    
    results['gaussian'] = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'activation_variances': activation_variances,
        'training_time': training_time
    }
    
    return results

# 运行所有实验
results = run_experiments()

# 可视化结果
plt.figure(figsize=(20, 15))

# 1. 比较训练和测试损失
plt.subplot(2, 3, 1)
for name, data in results.items():
    plt.plot(range(1, 21), data['test_losses'], label=name)
plt.title('Test Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 2. 比较训练和测试准确率
plt.subplot(2, 3, 2)
for name, data in results.items():
    plt.plot(range(1, 21), data['test_accuracies'], label=name)
plt.title('Test Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# 3. 比较无Dropout和有Dropout的激活值方差
plt.subplot(2, 3, 3)
for layer_name in results['no_dropout']['activation_variances'].keys():
    plt.plot(range(1, 21), results['no_dropout']['activation_variances'][layer_name], 
             label=f'无Dropout {layer_name}', linestyle='--')
    plt.plot(range(1, 21), results['dropout']['activation_variances'][layer_name], 
             label=f'有Dropout {layer_name}')
plt.title('Activation Variance Comparison')
plt.xlabel('Epoch')
plt.ylabel('Variance')
plt.legend()
plt.grid(True)

# 4. 比较所有方法的最终测试准确率
plt.subplot(2, 3, 4)
final_accuracies = {name: data['test_accuracies'][-1] for name, data in results.items()}
plt.bar(final_accuracies.keys(), final_accuracies.values())
plt.title('Final Test Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=45)
plt.grid(True)

# 5. 比较训练时间
plt.subplot(2, 3, 5)
training_times = {name: data['training_time'] for name, data in results.items()}
plt.bar(training_times.keys(), training_times.values())
plt.title('Training Time Comparison')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.savefig('all_experiments_results.png')
plt.show()

