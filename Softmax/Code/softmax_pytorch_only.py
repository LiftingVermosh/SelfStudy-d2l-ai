import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 下载数据集
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

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

class SoftmaxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(28*28, 10)  # 输入784像素, 输出10个类别
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear(x)
        return logits

model = SoftmaxModel()

# 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()  # 包含Softmax计算
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练参数
epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(epochs):
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        
        # 前向传播
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # 每轮结束计算准确率
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = correct / len(test_data)
    print(f"Epoch {epoch+1}: Accuracy={accuracy:.4f}, Test Loss={test_loss:.4f}")

classes = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

model.eval()
sample, label = test_data[0]
with torch.no_grad():
    pred = model(sample.unsqueeze(0).to(device))
    predicted = classes[pred.argmax(1).item()]
    actual = classes[label]
    
print(f"预测: '{predicted}', 实际: '{actual}'")
