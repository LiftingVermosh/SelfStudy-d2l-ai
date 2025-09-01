import torch
import torch.nn as nn

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

# 初始化卷积层
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 创建训练数据
X = torch.ones(6, 8)
X[:, 3:5] = 0  # 设置中间两列为0

# 目标卷积核为 [1, -1]
K_target = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K_target)

# 调整维度以适应卷积层
print(X)
X = X.reshape((1, 1, 6, 8))
print(X)
Y = Y.reshape((1, 1, 6, 7))

# 设置学习率和迭代次数
lr = 8e-3  
num_epochs = 100  

for epoch in range(num_epochs):
    # 前向传播
    Y_hat = conv2d(X)
    
    # 计算损失
    loss = ((Y_hat - Y) ** 2).sum()
    
    conv2d.zero_grad()  # 梯度清零

    # 反向传播
    loss.backward()

    # 更新权重
    with torch.no_grad():
        for param in conv2d.parameters():
            param -= lr * param.grad
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# 输出学习到的卷积核
learned_kernel = conv2d.weight.data.reshape(1, 2)
print(f'Learned kernel: {learned_kernel}')
print(f'Target kernel: {K_target}')
