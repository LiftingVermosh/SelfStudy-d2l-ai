
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class PolynomialRegression:
    def __init__(self, max_degree=20, n_train=100, n_test=100, normalize=True):
        self.max_degree = max_degree
        self.n_train = n_train
        self.n_test = n_test
        self.true_w = None
        self.features = None
        self.poly_features = None
        self.labels = None
        self.normalize = True
        self._generate_data()
    
    def _generate_data(self):
        """生成多项式回归的合成数据"""
        # 初始化真实权重（前min(4, max_degree)项非零）
        self.true_w = np.zeros(self.max_degree)
        n_nonzero = min(4, self.max_degree) 
        self.true_w[0:n_nonzero] = np.array([5, 1.2, -3.4, 2.5])[:n_nonzero]
        
        # 生成特征并多项式扩展
        features = np.random.normal(size=(self.n_train + self.n_test, 1))
        np.random.shuffle(features)
        poly_features = np.power(features, np.arange(self.max_degree).reshape(1, -1))
        
        # 归一化处理
        if self.normalize:  # 确保已添加normalize参数
            for i in range(self.max_degree):
                poly_features[:, i] /= math.gamma(i + 1)
        
        # 生成标签并添加噪声
        labels = np.dot(poly_features, self.true_w)
        labels += np.random.normal(scale=0.1, size=labels.shape)
        
        # 转换为PyTorch张量
        self.true_w = torch.tensor(self.true_w, dtype=torch.float32)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.poly_features = torch.tensor(poly_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def get_datasets(self):
        """返回训练集和测试集"""
        train_data = (self.poly_features[:self.n_train], 
                      self.labels[:self.n_train])
        test_data = (self.poly_features[self.n_train:], 
                     self.labels[self.n_train:])
        return train_data, test_data

class ModelTrainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.test_losses = []
        self.epoch_log = []
    
    def create_dataloader(self, features, labels, batch_size=10, shuffle=False):
        """创建数据加载器"""
        if features.dim() == 1:
            features = features.unsqueeze(1)
        dataset = TensorDataset(features, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def evaluate_loss(self, data_loader):
        """评估模型损失"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for X, y in data_loader:
                outputs = self.model(X)
                y = y.view_as(outputs)
                loss = self.loss_fn(outputs, y)
                total_loss += loss.item() * X.size(0)
                total_samples += X.size(0)
        
        return total_loss / total_samples
    
    def train_epoch(self, train_loader):
        """训练单个epoch"""
        self.model.train()
        epoch_loss = 0
        
        for X, y in train_loader:
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(X)
            y = y.view_as(outputs)
            
            # 损失计算和反向传播
            loss = self.loss_fn(outputs, y)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item() * X.size(0)
        
        return epoch_loss / len(train_loader.dataset)
    
    def train(self, train_data, test_data, num_epochs=400, batch_size=10):
        """完整的训练流程"""
        train_features, train_labels = train_data
        test_features, test_labels = test_data
        
        # 创建数据加载器
        train_loader = self.create_dataloader(
            train_features, train_labels, batch_size, shuffle=True
        )
        test_loader = self.create_dataloader(
            test_features, test_labels, batch_size
        )
        
        # 初始评估
        self.model.eval()
        with torch.no_grad():
            init_train_loss = self.evaluate_loss(train_loader)
            init_test_loss = self.evaluate_loss(test_loader)
        
        self.train_losses.append(init_train_loss)
        self.test_losses.append(init_test_loss)
        self.epoch_log.append(0)
        
        print(f'Training {num_epochs} epochs with batch size {batch_size}')
        print(f'Initial | Train Loss: {init_train_loss:.6f} | Test Loss: {init_test_loss:.6f}')
        
        # 训练循环
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            test_loss = self.evaluate_loss(test_loader)
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.epoch_log.append(epoch + 1)
            
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f'Epoch {epoch+1:3d}/{num_epochs} | '
                      f'Train Loss: {train_loss:.6f} | '
                      f'Test Loss: {test_loss:.6f}')
        
        # 最终预测
        with torch.no_grad():
            train_preds = self.model(train_features).numpy()
            test_preds = self.model(test_features).numpy()
        
        return train_preds, test_preds

def visualize_results(results, max_degree):
    """综合可视化报告"""
    plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2)
    
    # 损失曲线对比
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(results['epochs'], results['train_losses'], 'b-', label='Train Loss')
    ax1.plot(results['epochs'], results['test_losses'], 'r-', label='Test Loss')
    ax1.set_title('Training & Test Loss Curves')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.annotate(f'Final Test Loss: {results["test_losses"][-1]:.4f}', 
                 xy=(0.7, 0.9), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 权重对比
    ax2 = plt.subplot(gs[1, 0])
    width = 0.35
    indices = np.arange(len(results['true_w']))
    ax2.bar(indices - width/2, results['true_w'], width, label='True Weights', alpha=0.7)
    ax2.bar(indices + width/2, results['learned_w'], width, label='Learned Weights', alpha=0.7)
    ax2.set_title('Weight Comparison (True vs Learned)')
    ax2.set_xlabel('Feature Degree')
    ax2.set_ylabel('Weight Value')
    ax2.set_xticks(indices)
    ax2.set_xticklabels([f'x^{i}' for i in range(max_degree)])
    ax2.legend()
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # 预测值对比
    ax3 = plt.subplot(gs[1, 1])
    sorted_idx = np.argsort(results['test_features'][:, 0])
    ax3.scatter(results['test_features'][sorted_idx, 0], 
                results['test_labels'][sorted_idx], 
                s=20, label='True Values', alpha=0.6)
    ax3.plot(results['test_features'][sorted_idx, 0], 
             results['test_preds'][sorted_idx], 
             'r-', linewidth=2, label='Predictions')
    ax3.set_title('Test Set: True vs Predicted Values')
    ax3.set_xlabel('Feature Value')
    ax3.set_ylabel('Target Value')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('polynomial_regression_results.png', dpi=150)
    plt.show()

def main():
    # 数据准备
    max_degree = 20
    data_gen = PolynomialRegression(max_degree=max_degree)
    train_data, test_data = data_gen.get_datasets()
    train_features, train_labels = train_data
    test_features, test_labels = test_data
    
    # 模型初始化
    input_dim = train_features.shape[1]
    model = nn.Sequential(nn.Linear(input_dim, 1, bias=False))
    
    # 训练配置
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    trainer = ModelTrainer(model, loss_fn, optimizer)
    
    # 训练模型
    train_preds, test_preds = trainer.train(
        train_data, test_data, num_epochs=400
    )
    
    # 收集结果
    results = {
        'true_w': data_gen.true_w.numpy(),
        'learned_w': model[0].weight.detach().numpy().flatten(),
        'train_losses': trainer.train_losses,
        'test_losses': trainer.test_losses,
        'epochs': trainer.epoch_log,
        'train_features': train_features.numpy(),
        'train_labels': train_labels.numpy(),
        'train_preds': train_preds,
        'test_features': test_features.numpy(),
        'test_labels': test_labels.numpy(),
        'test_preds': test_preds
    }
    
    # 打印关键指标
    print("\n=== Training Summary ===")
    print(f"Final Train Loss: {results['train_losses'][-1]:.6f}")
    print(f"Final Test Loss: {results['test_losses'][-1]:.6f}")
    
    # 计算重要权量的误差
    significant_idx = np.where(np.abs(results['true_w']) > 0.1)[0]
    weight_errors = np.abs(results['true_w'][significant_idx] - results['learned_w'][significant_idx])
    print(f"\nSignificant Weight Errors:")
    for i, idx in enumerate(significant_idx):
        print(f"  x^{idx}: True={results['true_w'][idx]:.4f}, "
              f"Learned={results['learned_w'][idx]:.4f}, "
              f"Error={weight_errors[i]:.4f}")
    
    # 可视化结果
    visualize_results(results, max_degree)
    
    return results

def model_complexity_experiment():
    """ 模型复杂度实验 """
    degrees = range(1, 21)  # 测试1-20阶多项式
    train_losses, test_losses = [], []
    
    for degree in degrees:
        # 修改数据生成（限制多项式阶数）
        data_gen = PolynomialRegression(max_degree=degree)
        print(f"\nTraining degree {degree}...")
        train_data, test_data = data_gen.get_datasets()
        
        # 修改模型输入维度
        model = nn.Sequential(nn.Linear(degree, 1, bias=False))
        
        # 训练配置 - 不同阶数使用不同学习率
        lr = 0.01 if degree <= 10 else 0.001
        trainer = ModelTrainer(model, nn.MSELoss(), optim.SGD(model.parameters(), lr=lr))

        # 训练配置 - 减少epoch避免过拟合
        trainer = ModelTrainer(model, nn.MSELoss(), optim.SGD(model.parameters(), lr=0.01))
        trainer.train(train_data, test_data, num_epochs=200)  # 减少训练轮数
        
        # 记录最终损失
        train_losses.append(trainer.train_losses[-1])
        test_losses.append(trainer.test_losses[-1])

    # 非标准化训练
    train_losses_no_norm, test_losses_no_norm = [], []
    for degree in degrees:
        # 重新生成无归一化数据
        data_gen = PolynomialRegression(max_degree=degree, normalize=False)
        train_data, test_data = data_gen.get_datasets()
        
        # 创建新模型
        model = nn.Sequential(nn.Linear(degree, 1, bias=False))
        
        # 使用更稳定的优化器配置
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # 改用Adam
        trainer = ModelTrainer(model, nn.MSELoss(), optimizer)
        
        # 重新训练
        trainer.train(train_data, test_data, num_epochs=200)
        
        # 记录当前实验结果
        train_losses_no_norm.append(trainer.train_losses[-1])
        test_losses_no_norm.append(trainer.test_losses[-1])
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_losses, 'bo-', label='Train Loss')
    plt.plot(degrees, test_losses, 'rs-', label='Test Loss')
    plt.plot(degrees, train_losses_no_norm, 'c:', label='Train (No Norm)')
    plt.plot(degrees, test_losses_no_norm, 'm:', label='Test (No Norm)')
    plt.axvline(x=4, color='g', linestyle='--', label='Optimal Degree')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('degree_complexity.png')

def data_size_experiment():
    """ 数据规模实验 """
    sizes = [10, 20, 50, 100, 200, 500]  # 不同训练集大小
    train_losses, test_losses = [], []
    
    for size in sizes:
        # 修改数据生成规模
        data_gen = PolynomialRegression(n_train=size)
        print(f"\nTraining with {size} training examples...")
        train_data, test_data = data_gen.get_datasets()
        
        train_features = train_data[0][:, :4]
        test_features = test_data[0][:, :4]
        
        model = nn.Sequential(nn.Linear(4, 1, bias=False))
        trainer = ModelTrainer(model, nn.MSELoss(), optim.SGD(model.parameters(), lr=0.01))
        trainer.train((train_features, train_data[1]), (test_features, test_data[1]), num_epochs=400)
        
        train_losses.append(trainer.train_losses[-1])
        test_losses.append(trainer.test_losses[-1])
    
    # 绘制结果
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, train_losses, 'bo-', label='Train Loss')
    plt.plot(sizes, test_losses, 'rs-', label='Test Loss')
    plt.xscale('log')
    plt.xlabel('Training Set Size (log scale)')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('data_size_impact.png')


if __name__ == "__main__":
    # results = main()
    model_complexity_experiment()
    data_size_experiment()
