import torch
import torch.nn as nn
import numpy as np
import os
import sys
from Vocab import Vocab, numericalize
from torch.utils.data import DataLoader, TensorDataset
import data_preporcess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.training_visualizer import TrainingVisualizer


class RNNConfig:
    """配置类，集中管理所有超参数"""
    def __init__(self):
        self.batch_size = 32
        self.seq_len = np.random.randint(32, 64)        # 序列长度
        self.embed_size = 64    # 嵌入长度
        self.hidden_size = 128
        self.lr = 0.01
        self.epochs = 20  
        self.dropout_rate = 0.4
        self.weight_decay = 5e-4 
        self.train_split = 0.8
        self.min_freq = 2 
        self.reserved_tokens = ['<pad>', '<unk>', '<eos>']  


class RNNCell(nn.Module):
    """RNN基础单元"""
    def __init__(self, input_size, hidden_size):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size


        # 参数初始化
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, h):
        """前向传播"""
        h_next = torch.relu(self.layer_norm(x @ self.W_xh + h @ self.W_hh + self.b_h))
        return h_next

class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_size)
        self.rnn_cell = RNNCell(config.embed_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate) 
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.1, 0.1)

    def forward(self, x, h=None):
        batch_size, seq_len = x.shape
        
        if h is None:
            h = torch.zeros(batch_size, self.config.hidden_size, device=x.device)
        
        x_embed = self.embedding(x)
        x_embed = self.dropout(x_embed)  # 嵌入层后dropout
        
        h_seq = []
        for i in range(seq_len):
            h = self.rnn_cell(x_embed[:, i, :], h)
            h_seq.append(h)

        h_seq = torch.stack(h_seq, dim=1)
        h_seq = self.dropout(h_seq)  # RNN输出后dropout
        output = self.output_layer(h_seq)
        
        return output, h

class RNNDataProcessor:
    """数据处理类，封装数据加载和预处理逻辑"""
    def __init__(self, config):
        self.config = config
        self.vocab = None
        self.vocab_size = 0

    def load_and_preprocess_data(self, file_path):
        """加载和预处理数据"""
        # 加载原始数据
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        # 数据清洗和分词
        text = data_preporcess.data_clean_for_time_machine(file_content)
        tokens = data_preporcess.tokenize(text)
        
        # 构建词表
        self.vocab = Vocab(tokens, min_freq=self.config.min_freq, 
                          reserved_tokens=self.config.reserved_tokens)
        indices, self.vocab = numericalize(tokens, self.vocab)
        self.config.vocab_size = len(self.vocab)
        
        return torch.tensor(indices, dtype=torch.long)

    def create_sequences(self, data):
        """创建输入-目标序列对"""
        X, y = [], []
        for i in range(len(data) - self.config.seq_len):
            X.append(data[i:i + self.config.seq_len])
            y.append(data[i + 1:i + self.config.seq_len + 1])
        return torch.stack(X), torch.stack(y)

    def get_data_loaders(self, X, y):
        """创建训练和测试数据加载器"""
        train_size = int(self.config.train_split * len(X))
        
        train_dataset = TensorDataset(X[:train_size], y[:train_size])
        test_dataset = TensorDataset(X[train_size:], y[train_size:])
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, 
                                 shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, 
                                shuffle=False)
        
        return train_loader, test_loader


class RNNTrainer:
    """训练器类，封装训练逻辑"""
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=12
        )
        
    def train_epoch(self, train_loader, epoch):
        """单epoch训练"""
        self.model.train()
        self.config.seq_len = np.random.randint(32, 64)  # 序列长度随机变化
        total_loss = 0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            output, _ = self.model(X)
            loss = self.criterion(output.reshape(-1, self.config.vocab_size), 
                                 y.reshape(-1))
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(X)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        return total_loss / len(train_loader)

    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        test_loss = 0
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                output, _ = self.model(X)
                loss = self.criterion(output.reshape(-1, self.config.vocab_size), 
                                     y.reshape(-1))
                test_loss += loss.item() * X.size(0)
        
        return test_loss / len(test_loader.dataset)

    def train(self, train_loader, test_loader, visualizer=None):
        """完整训练流程"""
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            test_loss = self.evaluate(test_loader)

            self.scheduler.step(test_loss)
            
            print(f'Epoch {epoch}/{self.config.epochs}: '
                  f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
            
            if visualizer:
                visualizer.update_train(epoch, train_loss, 0)  # 准确率设为0，语言建模不计算准确率
                visualizer.update_test(epoch, test_loss, 0)


class TextGenerator:
    """文本生成器类"""
    def __init__(self, model, vocab, config, device):
        self.model = model
        self.vocab = vocab
        self.config = config
        self.device = device

    def generate(self, initial_seq, num_predict=10, temperature=1.0):
        """生成文本，支持温度参数控制随机性"""
        self.model.eval()
        
        with torch.no_grad():
            tokens = initial_seq.split()
            indices = [self.vocab.token_to_idx.get(token, 
                      self.vocab.token_to_idx['<unk>']) for token in tokens]
            
            # 序列长度处理
            if len(indices) < self.config.seq_len:
                pad_idx = self.vocab.token_to_idx['<pad>']
                indices.extend([pad_idx] * (self.config.seq_len - len(indices)))
            else:
                indices = indices[:self.config.seq_len]
            
            x = torch.tensor(indices, dtype=torch.long, 
                           device=self.device).unsqueeze(0)
            h = None
            
            print(f"初始序列: '{initial_seq}'")
            generated_tokens = []
            
            for i in range(num_predict):
                output, h = self.model(x, h)
                last_output = output[:, -1, :] / temperature
                prob = torch.softmax(last_output, dim=-1)
                next_idx = torch.multinomial(prob, 1).item()  # 带随机性的采样
                next_token = self.vocab.idx_to_token[next_idx]
                
                generated_tokens.append(next_token)
                print(f"预测第{i+1}个词: {next_token}")
                
                # 滑动窗口更新
                x = torch.cat([x[:, 1:], 
                             torch.tensor([[next_idx]], device=self.device)], dim=1)
            
            return ' '.join(generated_tokens)


def main():
    """主函数"""
    import time
    # 配置和设备设置
    config = RNNConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据预处理
    data_processor = RNNDataProcessor(config)
    data = data_processor.load_and_preprocess_data('./data/time-machine-data.txt')
    X, y = data_processor.create_sequences(data)
    train_loader, test_loader = data_processor.get_data_loaders(X, y)
    
    print(f'数据集大小: {X.shape}, 词表大小: {config.vocab_size}')
    
    # 模型初始化
    model = RNN(config).to(device)
    
    # 训练可视化
    visualizer = TrainingVisualizer()
    
    # 训练模型
    start_time = time.time()
    trainer = RNNTrainer(model, config, device)
    trainer.train(train_loader, test_loader, visualizer)
    end_time = time.time()
    print(f'训练用时: {end_time - start_time:.2f}s')
    
    # 文本生成
    generator = TextGenerator(model, data_processor.vocab, config, device)
    initial_seq = 'the time machine'
    generated_text = generator.generate(initial_seq, num_predict=10)
    
    print(f"\n完整生成结果: {initial_seq} {generated_text}")
    
    # 保存可视化结果
    target_path = '.\\Recurrnet Neural Network\\Models_Output'
    os.makedirs(target_path, exist_ok=True)
    visualizer.plot(os.path.join(target_path, f'RNN-TimeMachine_lr_{config.lr}_wd_{config.weight_decay}_dropout_{config.dropout_rate}.png'))


if __name__ == '__main__':
    main()
