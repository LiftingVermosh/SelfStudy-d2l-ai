import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np

def detect_covariate_shift_torch(X_train, X_test, input_dim, device='cpu', epochs=50, lr=0.01):
    """
    使用PyTorch神经网络二分类器检测协变量偏移。
    
    参数:
    X_train (np.ndarray): 训练集特征
    X_test (np.ndarray): 测试集特征
    input_dim (int): 输入特征的维度
    device (str): 'cpu' 或 'cuda'
    epochs (int): 训练轮数
    lr (float): 学习率
    
    返回:
    auc_score (float): 检测器的AUC分数
    """
    # 确保输入是numpy数组，然后转换为PyTorch张量
    if not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if not isinstance(X_test, np.ndarray):
        X_test = np.array(X_test)
    
    # 1. 准备数据和标签
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    
    # 合并特征，创建标签（0 for train, 1 for test）
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((np.zeros(n_train), np.ones(n_test)))
    
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X_combined).to(device)
    y_tensor = torch.FloatTensor(y_combined).to(device)
    
    # 创建数据集和数据加载器
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 2. 定义神经网络分类器
    class ShiftDetector(nn.Module):
        def __init__(self, input_size):
            super(ShiftDetector, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid() # 输出为0-1之间的概率
            )
        
        def forward(self, x):
            return self.net(x)
    
    # 初始化模型、损失函数和优化器
    model = ShiftDetector(input_dim).to(device)
    criterion = nn.BCELoss() # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 3. 训练模型
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}')
    
    # 4. 评估模型
    model.eval()
    with torch.no_grad():
        # 使用所有数据进行预测
        predictions = model(X_tensor).squeeze().cpu().numpy()
    
    # 计算AUC
    auc_score = roc_auc_score(y_combined, predictions)
    
    print(f"\n二分类协变量偏移检测器 AUC: {auc_score:.3f}")
    if auc_score > 0.65:
        print("警告：检测到显著的协变量偏移！")
    else:
        print("未检测到显著的协变量偏移。")
    
    return auc_score

# 使用示例
# 假设 train_features 和 test_features 是你的数据
# auc = detect_covariate_shift_torch(train_features, test_features, 
#                                   input_dim=train_features.shape[1], 
#                                   device='cuda' if torch.cuda.is_available() else 'cpu')
