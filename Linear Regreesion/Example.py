import torch
import random
import matplotlib.pyplot as plt
# from d2l import torch as d2l

def init_data(w, b, dataSize):
  # Initialize data for linear regression / 初始化线性回归数据
  X = torch.normal(0, 1, (dataSize, len(w))) # Normal distribution / 正态分布
  y = torch.matmul(X, w) + b # Linear equation / 线性方程
  y += torch.normal(0, 0.01, y.shape) # Add noise / 添加噪声
  return X, y

def data_iter(batch_size, features, labels):
  # Generate data iterators / 生成数据迭代器
  num_examples = len(features)
  indices = list(range(num_examples))
  random.shuffle(indices) # Shuffle indices / 打乱索引
  for i in range(0, num_examples, batch_size):
    j = torch.tensor(indices[i:min(i + batch_size, num_examples)])
    yield features[j], labels[j]  # Yield batch data,Stop iteration and waiting for next call / 生成批次数据，停止迭代并等待下次调用

true_w = torch.tensor([1.0, 1.2]) # slope / 斜率
true_b = torch.tensor(2.5)        # intercept / 截距
# Generate data / 生成数据
dataSize = 1000
features, labels = init_data(true_w, true_b, dataSize)

# Plot the original data / 绘制原始数据
plt.scatter(features[:, 0].numpy(), labels.numpy(), 1)
plt.show()

# Init model prameters / 初始化模型参数
w = torch.normal(0, 0.01, size=true_w.shape, requires_grad=True) # Initialize weights / 初始化权重
b = torch.zeros(1, requires_grad=True) # Initialize bias / 初始化偏置

# Define the model / 定义模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b  # Linear regression model / 线性回归模型

# Define the loss function / 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2  # Squared loss function / 平方损失函数

# Define the optimizer / 定义优化器
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size  # Update parameters / 更新参数
            param.grad.zero_()  # Reset gradients to zero / 梯度清零

# Training parameters / 训练参数
batch_size = 10  # Batch size / 批次大小
lr = 0.0001  # Learning rate / 学习率
num_epochs = 3  # Number of epochs / 迭代次数
net = linreg  # Model / 模型
loss = squared_loss  # Loss function / 损失函数

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size, features, labels):
       l = loss(net(X, w, b), y).sum() # Calculate loss / 计算损失
       l.sum().backward()  # Backpropagation / 反向传播
       sgd([w, b], lr, batch_size)  # Update parameters / 更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels).mean() # Calculate training loss / 计算训练损失
        print(f'epoch {epoch + 1}, loss {train_l:f}')  # Print training loss / 打印训练损失

# Print learned parameters / 打印学习到的参数
print(f'learned w: {w}, true w: {true_w},error w: {true_w - w}') 
print(f'learned b: {b}, true b: {true_b},error b: {true_b - b}')