import torch
import torch.nn as nn

def corr2d(X, K):
    """ 二维互相关运算 """
    h, w = K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()       # 截取窗口进行运算
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
Y = corr2d(X, K)
print(Y)

class Conv2d(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
    
# 黑白图片的边缘检测
if __name__ == '__main__':
    X = torch.ones(6, 8)
    X[:, 2:6] = 0
    print(X)
    K = torch.tensor([[1, -1]])
    Y = corr2d(X, K)
    print(Y)
    print(corr2d(X.t(), K))  # 转置卷积