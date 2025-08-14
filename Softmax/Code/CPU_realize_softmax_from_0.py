import torch
import img_classifation as ic
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 超参数优化
batch_size = 256
lr = 0.1  # 提高学习率加速收敛
num_epochs = 50  # 增加训练轮数
num_inputs = 784
num_outputs = 10

# 数据加载（添加归一化）
def load_data_normalized(batch_size):
    train_iter, test_iter = ic.load_data_fashion_mnist(batch_size)
    # 数据归一化 (0-255 -> 0-1)
    return ((X/255.0, y) for X,y in train_iter), ((X/255.0, y) for X,y in test_iter)

train_iter, test_iter = load_data_normalized(batch_size)

# 模型参数初始化（Xavier初始化）
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 数值稳定的Softmax实现（优化）
def stable_softmax(X):
    X_max = X.max(dim=1, keepdim=True).values.detach()  # 分离计算图
    X_exp = torch.exp(X - X_max)
    return X_exp / X_exp.sum(dim=1, keepdim=True)

# 网络定义（添加线性层）
def net(X):
    flattened = X.reshape(-1, num_inputs)
    linear = torch.matmul(flattened, W) + b
    return stable_softmax(linear)

# 交叉熵损失函数（向量化实现）
def cross_entropy(y_hat, y):
    log_probs = torch.log(y_hat[range(len(y_hat)), y])
    return -log_probs.mean()  # 直接返回标量损失

# 准确率计算（保持不变）
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

# 优化器重构（动量+SGD）
def sgd_updater_momentum(params, vs, lr, momentum=0.9):
    with torch.no_grad():
        for param, v in zip(params, vs):
            v[:] = momentum * v + lr * param.grad
            param -= v

# 训练循环重构
def train():
    # 初始化动量缓存
    v_W = torch.zeros_like(W)
    v_b = torch.zeros_like(b)
    
    train_losses, train_accs, test_accs = [], [], []
    
    for epoch in range(num_epochs):
        # 训练模式
        total_loss, total_acc, n = 0.0, 0.0, 0
        
        for X, y in train_iter:
            # 前向传播
            y_hat = net(X)
            
            # 损失计算
            loss = cross_entropy(y_hat, y)
            
            # 反向传播
            loss.backward()
            
            # 参数更新 (带动量)
            sgd_updater_momentum([W, b], [v_W, v_b], lr)
            
            # 记录指标
            total_loss += loss.item()
            total_acc += accuracy(y_hat, y)
            n += 1
        
        # 计算epoch指标
        avg_loss = total_loss / n
        avg_acc = total_acc / n
        test_acc = evaluate_accuracy(test_iter, net)
        
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Loss={avg_loss:.4f}, "
              f"Train Acc={avg_acc:.4f}, "
              f"Test Acc={test_acc:.4f}")
    
    print(f"\n最终测试准确率: {test_accs[-1]:.4f}")

# 评估函数（优化）
def evaluate_accuracy(data_iter, net):
    total_acc, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            total_acc += accuracy(y_hat, y)
            n += 1
    return total_acc / n

if __name__ == '__main__':
    train()
