import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
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

# 定义网络结构
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rates=[0.5, 0.2]):
        super(NeuralNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(hidden_sizes[1], output_size)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)
    
# 初始化模型、损失函数和优化器
model = NeuralNet(784, [500, 200], 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
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

# 训练和评估函数
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    model.train()
    train_losses = []
    train_accuracies = []
    test_losses = []  # 新增：存储测试损失
    test_accuracies = []
    
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
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, '
              f'Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%')
    
    return train_losses, train_accuracies, test_losses, test_accuracies

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
# 训练模型并计时
start_time = time.time()
train_losses, train_accuracies, test_losses, test_accuracies = train_model(
    model, train_loader, test_loader, criterion, optimizer, num_epochs=10
)
training_time = time.time() - start_time
print(f'训练时间: {training_time:.2f}秒')

# 可视化结果
plt.figure(figsize=(12, 8))
# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), train_losses, 'b-', label='Training Loss')
plt.plot(range(1, 11), test_losses, 'r-', label='Test Loss') 
plt.title('Training and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, 11), train_accuracies, 'b-', label='Training Accuracy')
plt.plot(range(1, 11), test_accuracies, 'r-', label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('(flip)Dropout_training_results_gpu.png')
plt.show()