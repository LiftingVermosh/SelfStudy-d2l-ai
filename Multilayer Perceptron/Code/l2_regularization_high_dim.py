import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 增加样本量 + 独立噪声特征
def generate_improved_data(num_samples=200, num_features=100):
    # 真实权重 (前5个有效)
    true_weights = np.zeros(num_features)
    true_weights[:5] = [2.0, -1.5, 3.0, 0.8, -2.2]
    
    # 生成独立特征（无相关性）
    X = np.random.randn(num_samples, num_features)
    
    # 增加目标噪声
    y = X[:, :5].dot(true_weights[:5]) + 1.5 * np.random.randn(num_samples)
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

X, y = generate_improved_data(num_samples=1200, num_features=100)

# 分割数据集
X_train, y_train = X[:len(X)//5], y[:len(y)//5]
X_test, y_test = X[4*len(X)//5:], y[4*len(y)//5:]

# 模型定义
class HighDimRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x).squeeze()

# 创建对比模型
input_dim = X.shape[1]
model_no_reg = HighDimRegression(input_dim)
model_with_reg = HighDimRegression(input_dim)
model_with_reg.load_state_dict(model_no_reg.state_dict())

# 正则化强度
criterion = nn.MSELoss()
# 超参数设置
optimizer_no_reg = torch.optim.SGD(model_no_reg.parameters(), lr=0.05, momentum=0.9)
optimizer_with_reg = torch.optim.SGD(
    model_with_reg.parameters(), 
    lr=0.05, 
    momentum=0.9, 
    weight_decay=0.1  # 增强正则化强度
)

# 训练循环（增加epoch）
losses_no_reg, losses_with_reg = [], []
epochs = 2000

for epoch in range(epochs):
    # 无正则化
    optimizer_no_reg.zero_grad()
    pred_no_reg = model_no_reg(X_train)
    loss_no_reg = criterion(pred_no_reg, y_train)
    loss_no_reg.backward()
    optimizer_no_reg.step()
    
    # 带正则化
    optimizer_with_reg.zero_grad()
    pred_with_reg = model_with_reg(X_train)
    loss_with_reg = criterion(pred_with_reg, y_train)
    loss_with_reg.backward()
    optimizer_with_reg.step()
    
    losses_no_reg.append(loss_no_reg.item())
    losses_with_reg.append(loss_with_reg.item())

# 评估函数
def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X)
        mse = criterion(pred, y).item()
        
        weights = model.linear.weight.data.numpy().flatten()
        eff_weights = weights[:5]  # 有效特征权重
        noise_weights = weights[5:]  # 噪声特征权重
        
        return mse, eff_weights, noise_weights

# 评估两个模型
train_mse_no_reg, eff_no_reg, noise_no_reg = evaluate_model(model_no_reg, X_train, y_train)
test_mse_no_reg, _, _ = evaluate_model(model_no_reg, X_test, y_test)

train_mse_with_reg, eff_with_reg, noise_with_reg = evaluate_model(model_with_reg, X_train, y_train)
test_mse_with_reg, _, _ = evaluate_model(model_with_reg, X_test, y_test)

# 计算噪声权重的L1范数（衡量稀疏性）
noise_l1_no_reg = np.abs(noise_no_reg).mean()
noise_l1_with_reg = np.abs(noise_with_reg).mean()

# 可视化结果
plt.figure(figsize=(15, 10))

# 1. 训练损失对比
plt.subplot(2, 2, 1)
plt.semilogy(losses_no_reg, label='No Regularization')
plt.semilogy(losses_with_reg, label='With L2 Regularization ')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.legend()
plt.title('Training Loss')

# 2. 有效特征权重对比
plt.subplot(2, 2, 2)
features = [f'F{i+1}' for i in range(5)]
true_weights = [2.0, -1.5, 3.0, 0.8, -2.2]

x = np.arange(len(features))
width = 0.25

plt.bar(x - width, true_weights, width, label='True Weights')
plt.bar(x, eff_no_reg, width, label='No Regularization')
plt.bar(x + width, eff_with_reg, width, label='With Regularization')
plt.xticks(x, features)
plt.ylabel('Weight Value')
plt.legend()
plt.title('Effective Feature Weights')

# 3. 噪声特征权重分布
plt.subplot(2, 2, 3)
plt.hist(noise_no_reg, bins=50, alpha=0.7, label='No Regularization')
plt.hist(noise_with_reg, bins=50, alpha=0.7, label='With Regularization')
plt.axvline(0, color='k', linestyle='--')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.legend()
plt.title('Noise Feature Weight Distribution')

# 4. 泛化性能对比
plt.subplot(2, 2, 4)
metrics = ['Train MSE', 'Test MSE']
x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, [train_mse_no_reg, test_mse_no_reg], width, label='No Reg')
plt.bar(x + width/2, [train_mse_with_reg, test_mse_with_reg], width, label='With Reg')
plt.xticks(x, metrics)
plt.ylabel('MSE')
plt.legend()
plt.title('Generalization Performance')

plt.tight_layout()
# plt.show()
plt.savefig('./Multilayer Perceptron/Output Figures/l2_regularization_high_dim.png')


# 打印关键指标
print(f"{'Metric':<25}{'No Regularization':>20}{'With Regularization':>20}")
print(f"{'Training MSE':<25}{train_mse_no_reg:>20.4f}{train_mse_with_reg:>20.4f}")
print(f"{'Test MSE':<25}{test_mse_no_reg:>20.4f}{test_mse_with_reg:>20.4f}")
print(f"{'Overfitting Gap':<25}{(test_mse_no_reg-train_mse_no_reg):>20.4f}{(test_mse_with_reg-train_mse_with_reg):>20.4f}")
print(f"{'Avg |Noise Weight|':<25}{noise_l1_no_reg:>20.4f}{noise_l1_with_reg:>20.4f}")
print(f"\n{'Effective Feature MAE:':<25}{np.abs(eff_no_reg - true_weights).mean():>20.4f}{np.abs(eff_with_reg - true_weights).mean():>20.4f}")
