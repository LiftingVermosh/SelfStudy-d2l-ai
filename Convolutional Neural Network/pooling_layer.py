import torch
import torch.nn.functional as F

def pooling_layer(X, kernel_size, pool_type='max', stride=1, padding=0):
    # 对输入X进行对称填充
    # F.pad参数：对于2D，填充格式为(left, right, top, bottom)
    X_padded = F.pad(X, (padding, padding, padding, padding))
    
    # 计算输出形状
    H_out = (X.shape[0] - kernel_size[0] + 2 * padding) // stride + 1
    W_out = (X.shape[1] - kernel_size[1] + 2 * padding) // stride + 1
    Y = torch.zeros((H_out, W_out))
    
    # 循环遍历输出位置的每个元素
    for i in range(H_out):
        for j in range(W_out):
            # 计算窗口的起始和结束索引
            h_start = i * stride
            h_end = h_start + kernel_size[0]
            w_start = j * stride
            w_end = w_start + kernel_size[1]
            
            # 从填充后的X中提取窗口
            window = X_padded[h_start:h_end, w_start:w_end]
            
            if pool_type == 'max':
                Y[i, j] = torch.max(window)
            elif pool_type == 'avg':
                Y[i, j] = torch.mean(window)
    return Y