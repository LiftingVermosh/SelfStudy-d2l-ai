import torch

def corr2d(X, K, padding=[0,0], stride=[1,1]):
    """ 二维互相关运算 """
    h, w = K.shape
    # padding 非零时，对X进行零填充
    if padding[0] > 0 or padding[1] > 0:
        X_padded = torch.zeros((X.shape[0] + 2 * padding[0], X.shape[1] + 2 * padding[1]))
        X_padded[padding[0]:padding[0] + X.shape[0], padding[1]:padding[1] + X.shape[1]] = X
        X = X_padded
    # 计算输出形状
    output_height = (X.shape[0] - h) // stride[0] + 1
    output_width = (X.shape[1] - w) // stride[1] + 1
    Y = torch.zeros((output_height, output_width))
    for i in range(output_height):
        for j in range(output_width):
            Y[i, j] = (X[i*stride[0]:i*stride[0]+h, j*stride[1]:j*stride[1]+w] * K).sum()
    return Y

if __name__ == '__main__':
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

