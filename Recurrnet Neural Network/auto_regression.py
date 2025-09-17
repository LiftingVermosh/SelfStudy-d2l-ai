import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 设置随机种子以确保可重复性
torch.manual_seed(42)

# 生成数据
length = 1000
t = torch.linspace(0, 10, length)  # 时间向量从0到10，1000个点
sin_data = torch.sin(t)  
noise = torch.normal(mean=0.0, std=0.2, size=(length,)) 
data = sin_data + noise

# 准备自回归数据
window_size = 64
inputs = []
targets = []
for i in range(window_size, length):
    inputs.append(data[i-window_size:i])  # 输入
    targets.append(data[i])  # 输出是下一个点

# 转换为PyTorch张量
inputs = torch.stack(inputs)  # 形状: [996, 4]
targets = torch.stack(targets).unsqueeze(1)  

# 创建数据集和DataLoader
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # 批量大小32，打乱数据

# 定义模型
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# 初始化模型
input_size = window_size  # 4
hidden_size = 10  # 自定义隐藏层大小，您可以根据需要调整
output_size = 1
model = SimpleMLP(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失，适用于回归问题
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam优化器，学习率0.01

# 训练循环（示例：训练10个epoch）
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    total_loss = 0
    for batch_inputs, batch_targets in dataloader:
        optimizer.zero_grad()  # 清零梯度
        outputs = model(batch_inputs)  # 前向传播
        loss = criterion(outputs, batch_targets)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

model.eval()  # 设置模型为评估模式
with torch.no_grad():
    predictions = model(inputs)  # 对所有输入进行预测
# 绘制预测结果和真实值
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
output_path = sys.path[0] + '/output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

plt.plot(t, data, label='Data')
plt.plot(t[window_size:], predictions, label='Predictions')
plt.legend()
# plt.show()
plt.savefig(f'{output_path}/predictions-window-size-{window_size}.png')