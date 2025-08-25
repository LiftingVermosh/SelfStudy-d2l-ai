import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 设置随机种子确保可重复性
torch.manual_seed(42)

# 1. 创建合成数据
def generate_data(num_samples=100):
    X = torch.randn(num_samples, 1) * 4  # 输入特征
    # 真实关系: y = 3x + 2 + 噪声
    y = 3 * X + 2 + torch.randn(num_samples, 1) * 1.5
    return X, y

X, y = generate_data()

# 2. 定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 单输入单输出
        
    def forward(self, x):
        return self.linear(x)

# 3. 创建两个相同初始化的模型
model_no_reg = LinearRegression()
model_with_reg = LinearRegression()
model_with_reg.load_state_dict(model_no_reg.state_dict())  # 确保相同初始权重

# 4. 设置训练参数
criterion = nn.MSELoss()
lr = 0.05
epochs = 200

# 创建优化器：一个不带L2，一个带L2正则化
optimizer_no_reg = torch.optim.SGD(model_no_reg.parameters(), lr=lr)
optimizer_with_reg = torch.optim.SGD(model_with_reg.parameters(), lr=lr, weight_decay=0.5)  # L2惩罚项

# 5. 训练循环
losses_no_reg, losses_with_reg = [], []
weights_no_reg, weights_with_reg = [], []

for epoch in range(epochs):
    # 无正则化模型
    optimizer_no_reg.zero_grad()
    pred_no_reg = model_no_reg(X)
    loss_no_reg = criterion(pred_no_reg, y)
    loss_no_reg.backward()
    optimizer_no_reg.step()
    
    # 带正则化模型
    optimizer_with_reg.zero_grad()
    pred_with_reg = model_with_reg(X)
    loss_with_reg = criterion(pred_with_reg, y)
    loss_with_reg.backward()
    optimizer_with_reg.step()
    
    # 记录数据和权重
    losses_no_reg.append(loss_no_reg.item())
    losses_with_reg.append(loss_with_reg.item())
    weights_no_reg.append(model_no_reg.linear.weight.item())
    weights_with_reg.append(model_with_reg.linear.weight.item())

# 6. 结果可视化
plt.figure(figsize=(15, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(losses_no_reg, label='No Regularization')
plt.plot(losses_with_reg, label='With L2 Regularization')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Comparison')

# 权重变化
plt.subplot(1, 2, 2)
plt.plot(weights_no_reg, label='No Regularization')
plt.plot(weights_with_reg, label='With L2 Regularization')
plt.axhline(y=3.0, color='r', linestyle='--', label='True Weight')
plt.xlabel('Epochs')
plt.ylabel('Weight Value')
plt.legend()
plt.title('Weight Value During Training')

plt.tight_layout()
# plt.show()
plt.savefig('Multilayer Perceptron\Output Figures\l2_regularization_low_dim.png')

# 7. 最终参数比较
print("\n最终参数对比:")
print(f"无正则化模型: weight = {model_no_reg.linear.weight.item():.4f}, bias = {model_no_reg.linear.bias.item():.4f}")
print(f"带L2正则化模型: weight = {model_with_reg.linear.weight.item():.4f}, bias = {model_with_reg.linear.bias.item():.4f}")
print(f"真实参数: weight = 3.0, bias = 2.0")
