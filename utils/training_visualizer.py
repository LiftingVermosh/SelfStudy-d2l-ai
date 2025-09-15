# training_visualizer.py
import os
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class TrainingVisualizer:
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
    
    def update_train(self, epoch, loss, acc):
        """更新训练数据"""
        self.epochs.append(epoch)
        self.train_losses.append(loss)
        self.train_accs.append(acc)
    
    def update_test(self, epoch, loss, acc):
        """更新测试数据"""
        # 确保测试数据与训练数据的 epoch 对应
        if epoch not in self.epochs:
            self.epochs.append(epoch)
        # 如果测试数据在训练数据之后更新，直接添加
        self.test_losses.append(loss)
        self.test_accs.append(acc)
    
    def plot(self, save_path=None):
        """绘制损失和准确率曲线"""
        plt.figure(figsize=(12, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, self.train_losses, label='Train Loss', marker='o')
        plt.plot(self.epochs, self.test_losses, label='Test Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.grid(True)
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.epochs, self.train_accs, label='Train Accuracy', marker='o')
        plt.plot(self.epochs, self.test_accs, label='Test Accuracy', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Test Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
