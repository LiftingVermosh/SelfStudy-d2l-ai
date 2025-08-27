# Chapter 5: Calculation of Deep Learning

## 5.1 块(Block)和层(Layer)

块(Block)是深度学习的基本组成单元，它由多个层(Layer)组成，每个层(Layer)又可以分为多个神经元(Neuron)组成。

层(Layer)的数量和类型决定了网络的复杂度和表达能力。

块可以理解为层的抽象表达，通过抽象，我们可以将复杂的网络拆分为多个层，通过递归的方式实现网络的构建，从而简化网络的设计。

d2l书中提到了编程视角下的块的概念，它被描述为类的实例

> `nn.Sequential`
> nn.Sequential 是 PyTorch 框架中的一种容器模块，它允许用户按照顺序将多个神经网络层或模块串联起来，从而构建一个完整的前向传播路径。这种方式使得模型定义更加直观和简洁。

### 5.1.1 自定义块

我们可以继承`nn.Module`类来自定义自己的块。

通常而言，块的基本功能包括：

1. Input as parameters - 将输入作为前向传播的参数
2. Output for backpropagation - 输出用于反向传播
3. store parameters - 存储参数
4. initialize parameters - 初始化参数

一个简单的自定义块的例子如下：

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, **kwargs):
        # 调用父类的初始化函数
        super(MLP, self).__init__(**kwargs)
        # 初始化其他参数 - 可选
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        # 定义前向传播
        return self.output(nn.ReLU()(self.hidden(x)))
```

这个自定义块`MLP`包含两个全连接层，前者的输入维度为20，输出维度为256，后者的输入维度为256，输出维度为10。前者使用ReLU激活函数，后者不使用激活函数。

我们可以实例化这个块，并用它作为神经网络的层

### 5.1.2 顺序块

`nn.Sequential`模块可以用来串联多个块。

```python
net = nn.Sequential(
    MLP(),
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10)
)
```

这个网络包含三个层：`MLP`块，`nn.Linear`层，`nn.ReLU`层。`MLP`块的输出作为`nn.Linear`层的输入，`nn.Linear`层的输出作为`nn.ReLU`层的输入，`nn.ReLU`层的输出作为`nn.Linear`层的输入，它们通过`nn.Sequential`模块串联起来。

当然，书上以自定义块梳理了顺序块的构建过程

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        for block in args:
            self.add_module(str(len(self)), block)

    def forward(self, input):
        for block in self._modules.values():
            input = block(input)
        return input
```

1. 继承`nn.Module`类
2. 定义`__init__`函数，接收任意数量的块作为参数，并将它们逐一添加到模块字典中
3. 定义`forward`函数，将输入通过每个块的`forward`函数，并将输出作为下一个块的输入

通过继承`nn.Module`类，我们可以方便地定义自己的块，并将它们串联起来，构建复杂的神经网络。

### Exercise 5.1.1

#### 1. 如果将MySequential中存储块的方式更改为Python列表，会出现什么样的问题？

此处题目的表述方式有一定误区，实际上，nn.Sequential 同时使用了列表和字典，但它们的角色不同。
    列表(或有序结构)：nn.Sequential 内部绝对维护着一个有序的模块序列。这是它最根本的功能——保证模块按照添加的顺序执行。在底层，它使用 OrderedDict(有序字典)来实现这一点，而不是普通的 dict。OrderedDict 会记住键值对插入的顺序，因此它兼具了列表的“有序性”和字典的“按key访问”的能力。
    字典(ModuleDict)：nn.Module(所有神经网络模块的基类，nn.Sequential 也继承自它)有一个机制：任何被赋值给类实例的子模块都会被自动注册(register)。
    当我们执行 self.linear = nn.Linear(20, 20) 或 self.net = nn.Sequential(...) 时，PyTorch 不仅仅是在创建一个Python属性。它还会将这个子模块(linear, net)添加到父模块(FixedHiddenMLP 或 NestMLP)内部的一个特殊字典(._modules)中。

回到问题本身：如果将 MySequential 中存储块的方式更改为 Python 列表，会出现什么样的问题？

- 参数无法被追踪：model.parameters() 将无法自动找到列表中的模块所包含的参数。此时我们必须手动编写代码去遍历列表并收集每个元素的参数，非常麻烦且容易出错。
- 无法正确保存和加载模型：model.state_dict() 将无法生成一个结构化的、带有有意义名称的状态字典。你可能只会得到一个庞大且无命名的一维参数列表，几乎无法用于后续的模型加载和微调。
- 无法移动到设备：model.to('cuda') 将失效，因为它无法找到列表中子模块的参数并将其移动到GPU上。
- 无法按名称访问和修改：你将失去通过字符串名字(如 print(model[0].net[1]))来访问、修改、甚至剪枝特定子模块的能力。调试和模型手术会变得异常困难。
- 破坏PyTorch的整个模块生态系统：nn.Sequential 本身也是一个 nn.Module。如果它内部的子模块不被注册，那么当你把 nn.Sequential 实例作为另一个大模块(如 NestMLP)的子模块时，这个 nn.Sequential 本身又成为了一个“黑盒”，它内部的参数又无法被它的父模块追踪到。这将导致整个模块化的设计彻底崩塌。
  
插句题外话：大量字典运行时的效率如何？

- 字典的查找速度比列表快，但字典的插入速度比列表慢。因此，如果我们需要在一个大列表中搜索或插入元素，那么使用字典会更快。
- 字典的内存占用比列表小，因此如果我们需要存储大量的模块，那么使用字典会更合适。
- 本末倒置：神经网络运行时前向传播是开销大头，99%以上的计算开销都花在执行张量运算(矩阵乘法、卷积、激活函数等)上。与在GPU上进行一次巨大的矩阵乘法相比，在CPU上进行几次字典查找的开销完全可以忽略不计，是纳秒(ns)级与毫秒(ms)级的差距。
- 模块数量有限：即使是非常深的网络(如ResNet-152)，其子模块的数量通常在几十到几百个的级别。对于现代CPU来说，在一个包含几百个键的字典中进行查找是极其高效的(O(1))。

#### 2. 实现一个块，它以两个块为参数，例如net1和net2，并返回前向传播中两个网络的串联输出。这也被称为平行块

```python
class Parallel(nn.Module):
    """
    平行块：将两个网络串联起来
    
    参数:
    - net1: 第一个网络
    - net2: 第二个网络
    """
    def __init__(self, net1, net2):
        super(Parallel, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x):
        # 假设沿着特征维度（dim=1）串联，确保其他维度相同
        return torch.cat(self.net1(x), self.net2(x), dim=1)
```

这个块接受两个网络作为参数，并在前向传播中串联它们的输出。

#### 3. 假设我们想要连接同一网络的多个实例。实现一个函数，该函数生成同一个块的多个实例，并在此基础上构建更大的网络

```python
import torch.nn as nn

class MultiInstanceNetwork(nn.Module):
    def __init__(self, block_class, num_instances, *block_args, **block_kwargs):
        """
        生成同一网络块的多个实例
        
        参数:
        - block_class: 要实例化的网络块类
        - num_instances: 要创建的实例数量
        - block_args: 传递给每个实例的位置参数
        - block_kwargs: 传递给每个实例的关键字参数
        """
        super(MultiInstanceNetwork, self).__init__()
        self.num_instances = num_instances
        
        # 使用ModuleList存储多个实例
        self.blocks = nn.ModuleList()
        for i in range(num_instances):
            self.blocks.append(block_class(*block_args, **block_kwargs))
    
    def forward(self, x):
        # 默认实现：依次通过每个实例
        for block in self.blocks:
            x = block(x)
        return x
```

这个函数接受一个块类、实例数量、位置参数和关键字参数，并生成一个包含多个块实例的网络。

当然我们也可以进行一些小小的拓展，比如支持更多的组合方式：

```python
class FlexibleMultiInstanceNetwork(nn.Module):
    def __init__(self, block_class, num_instances, combination_mode='sequential', 
                 *block_args, **block_kwargs):
        """
        生成同一网络块的多个实例，支持多种组合方式
        
        参数:
        - block_class: 要实例化的网络块类
        - num_instances: 要创建的实例数量
        - combination_mode: 组合方式，可选 'sequential', 'parallel', 'residual'
        - block_args: 传递给每个实例的位置参数
        - block_kwargs: 传递给每个实例的关键字参数
        """
    # 省略代码,与MultiInstanceNetwork一致
    
    def forward(self, x):
        if self.combination_mode == 'sequential':
            # 顺序连接：输入依次通过每个块
            for block in self.blocks:
                x = block(x)
            return x
        
        elif self.combination_mode == 'parallel':
            # 并行连接：输入同时通过所有块，输出求和
            outputs = []
            for block in self.blocks:
                outputs.append(block(x))
            return sum(outputs)  # 或者使用torch.cat进行拼接
        
        elif self.combination_mode == 'residual':
            # 残差连接：输入依次通过每个块，但添加跳跃连接
            identity = x
            for block in self.blocks:
                x = block(x)
            return x + identity  # 残差连接
        
        else:
            raise ValueError(f"不支持的组合模式: {self.combination_mode}")
```

## 5.2 参数访问

### 5.2.1 参数访问

对于一个块，书中提及可以通过索引来访问其中的参数。

```python
net = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

print(net[1].state_dict())  # 第二个层的状态字典
print(net[0].weight)  # 第一个全连接层的权重
print(net[2].bias)  # 第三个全连接层的偏置
```

此外，我们还可以通过`parameters()`函数来访问其所包含梯度的参数。

```python
net = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

print(list(net.parameters()))
```

但是，这种方式并不方便，我们更希望通过名字来访问参数。

```python
net = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

print(net.named_parameters())  # 按名称访问参数
```

`named_parameters()`函数返回一个元组列表，每个元组包含参数的名称和参数本身。

```python
for name, param in net.named_parameters():
    if name.endswith('weight'):
        print(name, param.size())
```

这个代码将打印出所有权重参数的名称和大小。

关于嵌套块，我们可以类比矩阵，将块视为矩阵的页，块内的层视为矩阵的列或者行，将参数视为矩阵的元素成员。

假设我们有一个块`net`包含两个子块`blk1`和`blk2`，`blk1`又包含两个子块`blk11`和`blk12`，`blk2`又包含两个子块`blk21`和`blk22`。

```python
net = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    ),
    nn.Linear(32, 10)
)
```

则在构建网络后，我们可以通过如下方式进行访问：

```python
    net[0][1][2].weight  # 第一个块的第二个子块的第三个层的权重
    net[1][2][0].bias  # 第二个块的第三个子块的第一个层的偏置
```

### 5.2.2 参数初始化

前文我们曾探讨过初始化的必要性，良好的初始化可以使得模型训练更加稳定，并减少训练时间。

PyTorch提供了多种初始化方法，包括常用的`nn.init.xavier_uniform_`和`nn.init.xavier_normal_`等。

```python
net = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# 初始化权重参数
for name, param in net.named_parameters():
    if 'weight' in name:
        nn.init.xavier_uniform_(param)

    elif 'bias' in name:
        nn.init.zeros_(param)
```

但我们也可以自定义初始化方法，考虑书上给出的模型：

$$w\sim\begin{cases}
    U(5,10)\quad proibility= \frac{1}{4}\\
    0\quad probability= \frac{1}{2}\\
    U(-10,-5)\quad probability= \frac{1}{4}
\end{cases}$$

考虑实现它：

```python
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(20, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = MyNet()

# 自定义初始化方法
def init_weights(m):
    if isinstance(m, nn.Linear):
        p = random.uniform(0, 1)
        if p < 0.25:
            nn.init.uniform_(m.weight, a=5, b=10)
        elif p < 0.75:
            nn.init.zeros_(m.weight)
        else:
            nn.init.uniform_(m.weight, a=-10, b=-5)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# 应用初始化方法
net.apply(init_weights)
```

### 5.2.3 共享参数

有时我们希望多个层使用相同的参数，这时我们可以将参数共享。共享的操作很简单，多次调用一个层实例即可

```python
import torch.nn as nn

# 创建一个共享的线性层
shared_fc = nn.Linear(10, 10)

# 在Sequential中多次使用这个共享层
model = nn.Sequential(
    nn.Linear(20, 10),
    nn.ReLU(),
    shared_fc,
    nn.ReLU(),
    shared_fc,  # 再次使用同一个实例
    nn.ReLU(),
    nn.Linear(10, 5)
)

# 打印模型结构
print(model)

# 检查两个共享层是否是同一个对象
print(model[2] is model[4])  # 输出: True

# 检查参数是否共享
for name, param in model.named_parameters():
    print(name, param.size())
```

由于共享，这个模型只有4组参数（包括偏置），而不是5组（如果两个共享层是独立的，则会多一组参数）。

### Exercises 5.2

#### 为什么共享参数是个好主意?

简单来说，**共享参数的核心思想是：让模型的多个部分使用同一组参数（权重和偏置）**。这不仅是为了节省内存，更深层的意义在于**强制模型学习到一种通用的、可重复使用的特征表示**，具体而言：

1. 显著减少模型参数数量，降低过拟合风险，这是最直观的好处。

  - 现代神经网络动辄拥有数百万甚至数十亿个参数。参数越多，模型的**容量**（拟合能力）就越大，但也越容易记住训练数据中的噪声而非学习其潜在规律，导致**过拟合**。
  - 通过共享参数，我们可以用极少的参数定义一个非常深的或重复的模型结构。

2. 实现“平移不变性”或“时序不变性”（核心归纳偏置）

这是共享参数最重要、最根本的原因。它赋予了模型一个先验知识，即“**在这个领域，学到的模式应该在任何位置都适用**”。

*   **卷积神经网络 (CNN)**：
    *   **直觉**：一张图片中的“猫耳朵”特征，无论它出现在左上角还是右下角，都应该是同一个特征。我们不应该为每个位置学习一个独立的“猫耳朵探测器”。
    *   **实现**：CNN中的**卷积核**就是参数共享的完美体现。同一个卷积核（一组参数）会滑过整个输入图像，在不同的位置检测**相同的模式**。这强制模型学习**平移不变**的特征。
    *   **没有共享会怎样？** 如果每个像素位置都有独立的参数，模型将无法识别出现在新位置的相同物体，泛化能力极差，且需要海量的训练数据。

*   **循环神经网络 (RNN) / Transformer**:
    *   **直觉**：在自然语言中，“not good”和“not bad”中的“not”所表达的否定含义，无论它出现在句子的开头、中间还是结尾，其语法功能是相同的。
    *   **实现**：RNN在每个时间步使用**相同的**循环单元（如LSTM或GRU细胞）来处理序列中的每个词。这强制模型学习**时序不变**的特征，即一个单词或短语的功能与其在序列中的绝对位置无关。
    *   **Transformer**中的自注意力机制虽然计算方式不同，但其核心的 `Q, K, V` 投影矩阵也是在整个序列上共享的，实现了类似的效果。

3. 使得模型能够处理可变长度的输入

这是参数共享带来的一个关键能力。

*   **问题**：我们训练的模型需要能处理不同长度的问题，例如翻译不同长度的句子、分析不同时间长度的音频等。
*   **解决方案**：通过共享参数，模型可以被看作一个**循环**或**递归**的程序。同一组参数被反复应用，无论输入有多长。
*   **例子**：训练好的RNN模型，可以用它来处理一个10个词的句子，也可以处理一个50个词的句子。我们不需要为每种可能的输入长度训练一个全新的模型。模型的**能力是通用的**。

4. 更高效的训练和推理

*   **更少的参数**意味着：
    *   **更小的内存占用**：模型更容易在内存有限的设备（如手机）上部署。
    *   **更快的计算速度**：需要计算和更新的参数更少。
    *   **更低的显存需求**：训练时所需的GPU显存更少。
    *   **更快的收敛**：需要优化的参数空间更小，有时可以帮助模型更快地找到好的解。

## 5.3 延后初始化

顾名思义，延后初始化（Deferred Initialization）是指在模型创建后，再初始化模型参数。

### Exercises 5.3

####  如果指定了第一层的输入尺寸，但没有指定后续层的尺寸，会发生什么？是否立即进行初始化？

事实上这一做法是标准的，PyTorch的层是独立定义的。每一层只需要知道自己的输入和输出尺寸。它不关心也不应该关心上一层的输出尺寸是多少，因为PyTorch的动态图机制会在运行时自动处理张量的流动。但不会立即进行初始化，当我们执行 `self.fc1 = nn.Linear(20, 256)` 时，发生的是：
- `PyTorch` 创建了一个 `Linear` 模块对象。
- 它为权重（ `weight` ）和偏置（ `bias` ）参数分配了内存空间，但这些参数的值是未初始化的（即它们是垃圾值，不是真正的权重）。
- 真正的、有意义的初始化（例如Kaiming初始化、Xavier初始化或自定义初始化）是在调用 `net.apply(init_func)` 或第一次执行前向传播 `forward(x)` 时完成的。

####  如果指定了不匹配的维度会发生什么？

错误不会在模型定义时抛出，而是在尝试进行前向传播（ `forward`）时 `PyTorch` 抛出一个 `RuntimeError` 异常，指出维度不匹配。

####  如果输入具有不同的维度，需要做什么？

- 最简单的就是固定输入尺寸并进行检查
- 也可以修改模型架构以适应动态输入，即加入自适应池化层

## 5.4 自定义层

继承 `nn.Module` 类并实现 `forward` 方法，即可定义自己的层。

```python
import torch.nn as nn

class MyLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return x @ self.weight.t() + self.bias
```

上述代码自定义了一个权重和偏置服从正态分布的全连接层。

### Exercises 5.4

####  设计一个接受输入并计算张量降维的层，它返回 $y_k=\sum_{i,j}W_{ijk}x_ix_j$

```python
import torch
import torch.nn as nn

class QuadraticLayer(nn.Module):
    def __init__(self, in_features, out_features):
        """
        初始化二次层。
        参数:
            in_features (int): 输入特征数。
            out_features (int): 输出特征数。
        """
        super(QuadraticLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 初始化权重张量 W，形状为 (in_features, in_features, out_features)
        self.weight = nn.Parameter(torch.randn(in_features, in_features, out_features))

    def forward(self, x):
        """
        前向传播计算 y_k = sum_{i,j} W_{ijk} x_i x_j。
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        # 使用 einsum 计算二次形式: y_{b,k} = sum_{i,j} x_{b,i} x_{b,j} W_{i,j,k}
        return torch.einsum('bi,bj,ijk->bk', x, x, self.weight)

```

代码通过 `torch.einsum` 函数计算二次形式，并返回输出张量

> **einsum（Einstein summation convention，爱因斯坦求和约定）**
> einsum是 PyTorch、NumPy、TensorFlow 等科学计算库中一个极其强大和通用的函数。它提供了一种简洁、可读的方式来表达多维数组（张量）的复杂线性代数运算。
> 其核心思想是：当一个下标在乘积项的同一部分重复出现时，就默认对这个下标的所有可能值进行求和，并省略求和符号 $\sum$
>  `Pytorch` 中 `einsum` 的语法为：
> `torch.einsum('subscripts_string', operand1, operand2, ...)`
> 其中 `subscripts_string` 为下标字符串，`operand1`、`operand2` 等为张量，`subscripts_string` 解读如下：
> - `b`：表示批次维度
> - `i`、`j`、`k`：表示输入张量的维度
> - `k`：表示输出张量的维度
>
> 例如，`torch.einsum('bi,bj,ijk->bk', x, x, self.weight)` 表示对输入张量 `x` 进行二阶求和，输出张量的维度为 `b` 和 `k`，输入张量的维度为 `i` 和 `j`，权重张量的维度为 `i`、`j` 和 `k`。
>
> 更具体地说，它的过程为：
> 定义输入：
>   - 'bi'：第一个输入 x，是一个二维张量。b 代表 batch 维度，i 代表特征维度。
>   - 'bj'：第二个输入也是 x（同一个张量用了两次）。b 还是 batch，j 是另一个特征维度标签（与 i 大小相同）。
>   - 'ijk'：第三个输入 W，是一个三维张量。i 和 j 是特征维度（必须与 x 的 i, j 大小匹配），k 是输出的特征维度。
>
>定义输出：'->bk'。
>   - 箭头 -> 右边定义了输出的维度标签。
>   - 注意到 i 和 j 没有出现在输出标签中。根据爱因斯坦约定，所有没有在输出中出现的标签，都会被求和消去。
>   - 所以，这个操作的含义是：对 i 和 j 进行求和，保留 b（批次）和 k（输出特征）维度。
>
> 执行过程：
>   - 它先计算 x 和 x 的外积（对于每个样本），得到一个临时张量，形状为 [b, i, j]。
>   - 然后将这个临时张量与权重 W [i, j, k] 进行逐元素相乘。
>   - 最后对消去的维度 i 和 j 求和，得到形状为 [b, k] 的输出。

#### 设计一个返回输入数据的傅立叶系数前半部分的层

```python
import torch
import torch.nn as nn

class FourierCoefficientLayer(nn.Module):
    def __init__(self, norm='ortho'):
        """
        初始化傅立叶系数层
        参数:
            norm (str): 归一化模式，可选 'backward', 'ortho' 或 'forward'
        """
        super(FourierCoefficientLayer, self).__init__()
        self.norm = norm

    def forward(self, x):
        """
        前向传播：计算输入数据的傅立叶系数前半部分
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length)
        返回:
            torch.Tensor: 傅立叶系数前半部分，形状为 (batch_size, n_coefficients)
        """
        # 确保输入是二维的 (batch_size, sequence_length)
        if x.dim() != 2:
            raise ValueError("输入应为二维张量 (batch_size, sequence_length)")
        # 计算快速傅立叶变换(FFT)
        # 对于实数输入，rfft只返回非负频率部分，正好是我们需要的前半部分
        fourier_coeffs = torch.fft.rfft(x, dim=1, norm=self.norm)
        return fourier_coeffs
```
