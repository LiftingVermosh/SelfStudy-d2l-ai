import torch
import torchvision
from torchvision import transforms
from d2l import torch as d2l
import os
import time
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.manual_seed(42)  # 固定随机种子

# 检测可用设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 超参数设置 
batch_size = 256
lr = 0.1  # 学习率
num_epochs = 20
num_inputs = 784
num_outputs = 10

# 数据预处理
def load_data_fashion_mnist(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # FashionMNIST的标准归一化
    ])
    
    train_set = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform)
    
    num_workers = 0 if os.name == 'nt' else 4
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size, shuffle=True, 
        pin_memory=True, num_workers=num_workers)
    
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size, shuffle=False,
        pin_memory=True, num_workers=num_workers)
    
    return train_loader, test_loader

# 加载数据
print("加载FashionMNIST数据集...")
train_iter, test_iter = load_data_fashion_mnist(batch_size)
print(f"训练批次: {len(train_iter)}, 测试批次: {len(test_iter)}")

# 模型参数初始化 - 修复3：调整初始化规模
print("初始化模型参数...")
W = torch.normal(0, 0.1, size=(num_inputs, num_outputs),  # 增大标准差
                 device=device, requires_grad=True)
b = torch.zeros(num_outputs, device=device, requires_grad=True)

# 网络定义
def net(X):
    X = X.to(device)
    flattened = X.reshape(-1, num_inputs)
    logits = torch.matmul(flattened, W) + b
    return logits

# 损失函数 
def stable_cross_entropy(y_hat, y):
    """数值稳定的交叉熵损失实现"""
    # 减去最大值提高数值稳定性
    logits_max = y_hat.max(dim=1, keepdim=True)[0]
    logits_shifted = y_hat - logits_max
    exp_logits = torch.exp(logits_shifted)
    sum_exp = exp_logits.sum(dim=1, keepdim=True)
    log_probs = logits_shifted - torch.log(sum_exp)
    
    # 只取正确类别的log概率
    nll = -log_probs[range(len(y)), y]
    return nll.mean()

# 准确率计算
def accuracy(y_hat, y):
    y = y.to(device)
    return (y_hat.argmax(dim=1) == y).float().mean().item()

# 累加器类
class Accumulator: 
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def __getitem__(self, idx):
        return self.data[idx]

# 评估函数
def evaluate_accuracy(data_iter, net):
    metric = Accumulator(2)  # 正确样本数, 总样本数
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            metric.add(accuracy(y_hat, y) * y.size(0), y.size(0))
    return metric[0] / metric[1]

# 训练一个epoch 
def train_epoch(net, train_iter, loss_fn, optimizer, epoch):
    metric = Accumulator(3)  # 损失总和, 准确率总和, 样本数
    total_batches = len(train_iter)
    
    start_time = time.time()
    for i, (X, y) in enumerate(train_iter):
        X, y = X.to(device), y.to(device)
        
        # 前向传播
        logits = net(X)
        loss = loss_fn(logits, y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # 确保使用优化器更新参数
        
        # 计算指标
        with torch.no_grad():
            acc = accuracy(logits, y)
            metric.add(loss.item() * y.size(0), acc * y.size(0), y.size(0))
        
        # 每100批次打印进度
        if i % 100 == 0 or i == total_batches - 1:
            batch_loss = metric[0] / metric[2]
            batch_acc = metric[1] / metric[2]
            print(f"Epoch {epoch+1} | Batch {i+1}/{total_batches} | "
                  f"Loss: {batch_loss:.4f} | Acc: {batch_acc:.4f}")
    
    epoch_time = time.time() - start_time
    train_loss = metric[0] / metric[2]
    train_acc = metric[1] / metric[2]
    return train_loss, train_acc, epoch_time

# 创建优化器
def create_optimizer(params, lr):
    return torch.optim.SGD(params, lr=lr)

# 主训练函数
def train(net, train_iter, test_iter, num_epochs):
    # 创建优化器 
    optimizer = create_optimizer([W, b], lr)
    
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1.0],
                            legend=['train loss', 'train acc', 'test acc'])
    
    best_acc = 0.0
    print(f"开始GPU加速训练 ({device})...")
    
    for epoch in range(num_epochs):
        train_loss, train_acc, epoch_time = train_epoch(
            net, train_iter, stable_cross_entropy, optimizer, epoch)  # 使用自定义损失
        
        test_acc = evaluate_accuracy(test_iter, net)
        
        # 更新动画
        animator.add(epoch + 1, (train_loss, train_acc, test_acc))
        
        # 打印epoch结果
        # print(f"\nEpoch {epoch+1}/{num_epochs} | "
        #       f"Time: {epoch_time:.1f}s | LR: {lr:.6f}")
        # print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        # print(f"  Test Acc: {test_acc:.4f}\n")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"新最佳测试准确率: {best_acc:.4f}")
    
    print("\n训练完成!")
    return best_acc

# 主程序
if __name__ == '__main__':
    # 开始训练
    best_acc = train(net, train_iter, test_iter, num_epochs)
    print(f"最终最佳测试准确率: {best_acc:.4f}")
    
    # 保存模型参数
    torch.save({'W': W, 'b': b}, 'softmax_model.pth')
    print("模型参数已保存")
    
    # 验证参数
    print(f"\n权重矩阵形状: {W.shape} | 偏置向量形状: {b.shape}")
    print(f"权重均值: {W.mean().item():.4f} | 权重标准差: {W.std().item():.4f}")
    print(f"偏置均值: {b.mean().item():.4f}")
    
    # 测试单个样本
    test_sample, test_label = next(iter(test_iter))
    test_sample = test_sample[0:1].to(device)
    with torch.no_grad():
        logits = net(test_sample)
        probs = torch.nn.functional.softmax(logits, dim=1)
        predicted = probs.argmax(dim=1).item()
    print(f"\n测试样本预测: {predicted} (实际: {test_label[0].item()})")
    print(f"预测概率分布: {probs.cpu().numpy().round(3)}")
