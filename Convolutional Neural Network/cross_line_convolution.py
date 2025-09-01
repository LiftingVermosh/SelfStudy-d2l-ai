import torch 
import torch.nn as nn 

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 构建对角线边缘图像
X = torch.ones((6, 6))
for i in range(6):
    X[i, i] = 0

# 构建卷积核并计算特征
print(f'X=\n{X}')
K = torch.tensor([[1, -1]])
Y = corr2d(X, K)
print(f'Y=\n{Y}')
Y_transport_k = corr2d(X, K.t())
print(f'Y_transport_k=\n{Y_transport_k}')

