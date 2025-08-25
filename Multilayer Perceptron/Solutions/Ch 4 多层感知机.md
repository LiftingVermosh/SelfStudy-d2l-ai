# Chapter 4 多层感知机

## 4.1 多层感知机

### 1. 计算pReLU激活函数的导数

$$
pReLU(x) = \max(0,x) + a * \min(0,x) \\
\frac{\partial pReLU(x)}{\partial x} = \begin{cases} 1, & x \geq 0 \\ a, & x < 0 \end{cases}
$$

### 2. 证明一个仅使用ReLU（或pReLU）的多层感知机构造了一个连续的分段线性函数

$$
\begin{cases}H = W_1X+b \\
Y = \text{ReLU}(H)W_2+c \\
ReLU(x) = \max(0,x)
\end{cases}
\Rightarrow
\begin{cases} Y = c \quad \text{if } H \leq 0 \\
Y = W_2\text{ReLU}(H) + c=W_1W_2X + W_2b + c \quad \text{otherwise}
\end{cases}
$$

因此，多层感知机的每一层都是一个连续的分段线性函数，因此它是一个连续的非线性函数。

### 3. 证明tanh(x) + 1 = 2 sigmoid(2x)

$$
\tanh(x) + 1 = \frac{e^x-e^{-x}}{e^x+e^{-x}} + 1 = \frac{2}{1+e^{-2x}} = 2\sigma(2x)
$$

### 4. 假设我们有一个非线性单元，将它一次应用于一个小批量的数据。这会导致什么样的问题

这会导致梯度消失或爆炸。原因是，非线性单元的输出会导致梯度的指数级增长，这会导致梯度爆炸。解决方法是使用合适的激活函数，如ReLU或pReLU

---

## 4.2 多层感知机的从零开始实现

### 1. 在所有其他参数保持不变的情况下，更改超参数num_hiddens的值，并查看此超参数的变化对结果有何影响。确定此超参数的最佳值

### 2. 尝试添加更多的隐藏层，并查看它对结果有何影响

### 3. 改变学习速率会如何影响结果？保持模型架构和其他超参数（包括轮数）不变，学习率设置为多少会带来最好的结果？

过低的学习率导致模型收敛过慢，过高的学习率导致模型无法收敛。

从测试结果而言，lr < 0.1 时存在收敛过慢的问题，lr > 1时存在无法收敛的问题。

### 4. 通过对所有超参数（学习率、轮数、隐藏层数、每层的隐藏单元数）进行联合优化，可以得到的最佳结果是什么？

### 5. 描述为什么涉及多个超参数更具挑战性

超参数的组合数随超参数的个数上升呈指数级增长，因此需要有更完善的搜索策略来找到最佳超参数组合。

### 6. 如果想要构建多个超参数的搜索方法，请想出一个聪明的策略

随机搜索、遗传算法、模拟退火算法、贝叶斯优化算法等.

---

## 4.3 多层感知机的简洁实现

### 1. 尝试添加不同数量的隐藏层（也可以修改学习率），怎么样设置效果最好？

- 隐藏层数量：通常1-2层即可，层数过多可能不会带来明显提升，反而增加计算量和过拟合风险。
- 隐藏层节点数：每层256到512个节点是常用的选择。
- 学习率：对于SGD，学习率在0.01到0.1之间比较常见。但具体需要根据训练情况调整。如果使用Adam优化器，则可以使用0.001的默认学习率。

### 2. 尝试不同的激活函数，哪个效果最好？

- ReLU (Rectified Linear Unit)：最常用，计算简单，梯度在正区间为1，训练速度快。缺点是在负数区域梯度为0（“死亡ReLU”问题）。在MLP中表现通常很好。
- LeakyReLU：解决了ReLU的死亡问题，在负数区域有一个小的斜率（如0.01）。有时比ReLU效果稍好，但并不是绝对的。
- Sigmoid：在隐藏层中不推荐使用，因为容易导致梯度消失，尤其是在深层网络中。
- Tanh：输出范围为(-1,1)，在隐藏层中有时被使用，但通常效果不如ReLU，因为梯度在绝对值较大时也会消失。

### 3. 尝试不同的方案来初始化权重，什么方法效果最好？

- 随机初始化：如从均值为0、标准差为0.01的正态分布中采样。这是代码中使用的，但标准差的设置很关键。如果设置过小，可能导致前向传播信号消失；过大则可能造成梯度爆炸。
- Xavier初始化：适用于使用Tanh或Sigmoid激活函数的网络。它根据输入和输出的维度来调整方差，保持各层输出的方差稳定。具体分为均匀分布和正态分布两种。
- He初始化：专为ReLU及其变种设计。它使用均值为0，标准差为sqrt(2/fan_in)的正态分布，其中fan_in是权重输入端的神经元数量。这种方法在ReLU网络中表现最好。

---

## 4.4 过拟合与欠拟合

### 1. 这个多项式回归问题可以准确地解出吗？提示：使用线性代数

对于任意阶数的多项式，由于normal equation的存在，可以用矩阵求逆法求解，其解存在且唯一，因此，对于多项式的回归理论上是可以准确解出的。但由于噪音的存在，我们在足够样本的情况下只能得到一个比较好的近似解。

normal equation的解法：

$$
\begin{bmatrix}
\frac{\partial}{\partial w_j} \frac{1}{2} \sum_{i=1}^n (y_i - \hat{y}_i)^2 \\
\frac{\partial}{\partial w_0} \frac{1}{2} \sum_{i=1}^n (y_i - \hat{y}_i)^2
\end{bmatrix} = 0
\\
\begin{bmatrix}
\sum_{i=1}^n (y_i - \hat{y}_i) x_i \\
\sum_{i=1}^n (y_i - \hat{y}_i)
\end{bmatrix} = 0
\\
w = (X^TX)^{-1}X^Ty
$$

### 2. 考虑多项式的模型选择

1. 绘制训练损失与模型复杂度（多项式的阶数）的关系图。观察到了什么？需要多少阶的多项式才
能将训练损失减少到0?
2. 在这种情况下绘制测试的损失图。
3. 生成同样的图，作为数据量的函数。

图见 `Output Figures` 文件夹，给出训练报告：

```bash
python '.\Multilayer Perceptron\Code\Overfitting_And_Underfitting.py'
erceptron\Code\Overfitting_And_Underfitting.py'                               
Training degree 1...
Training 200 epochs with batch size 10
Initial | Train Loss: 32.795579 | Test Loss: 32.943511
Epoch  1/200 | Train Loss: 27.528705 | Test Loss: 22.016845
Epoch 50/200 | Train Loss: 0.010865 | Test Loss: 0.009516
Epoch 100/200 | Train Loss: 0.010868 | Test Loss: 0.009514
Epoch 150/200 | Train Loss: 0.010861 | Test Loss: 0.009522
Epoch 200/200 | Train Loss: 0.010865 | Test Loss: 0.009519

Training degree 2...
Training 200 epochs with batch size 10
Initial | Train Loss: 28.639135 | Test Loss: 30.056591
Epoch  1/200 | Train Loss: 24.209834 | Test Loss: 20.403486
Epoch 50/200 | Train Loss: 0.009196 | Test Loss: 0.013862
Epoch 100/200 | Train Loss: 0.009197 | Test Loss: 0.013852
Epoch 150/200 | Train Loss: 0.009189 | Test Loss: 0.013852
Epoch 200/200 | Train Loss: 0.009180 | Test Loss: 0.013840

Training degree 3...
Training 200 epochs with batch size 10
Initial | Train Loss: 17.160746 | Test Loss: 13.817294
Epoch  1/200 | Train Loss: 15.262716 | Test Loss: 10.518249
Epoch 50/200 | Train Loss: 0.031242 | Test Loss: 0.045024
Epoch 100/200 | Train Loss: 0.010773 | Test Loss: 0.010802
Epoch 150/200 | Train Loss: 0.010725 | Test Loss: 0.010447
Epoch 200/200 | Train Loss: 0.010733 | Test Loss: 0.010435

Training degree 4...
Training 200 epochs with batch size 10
Initial | Train Loss: 20.949752 | Test Loss: 21.055416
Epoch  1/200 | Train Loss: 16.655872 | Test Loss: 13.800238
Epoch 50/200 | Train Loss: 0.436746 | Test Loss: 0.446558
Epoch 100/200 | Train Loss: 0.133536 | Test Loss: 0.129851
Epoch 150/200 | Train Loss: 0.044793 | Test Loss: 0.044827
Epoch 200/200 | Train Loss: 0.018845 | Test Loss: 0.020296

Training degree 5...
Training 200 epochs with batch size 10
Initial | Train Loss: 23.834523 | Test Loss: 20.682762
Epoch  1/200 | Train Loss: 21.716658 | Test Loss: 15.567698
Epoch 50/200 | Train Loss: 0.084247 | Test Loss: 0.072248
Epoch 100/200 | Train Loss: 0.052413 | Test Loss: 0.042874
Epoch 150/200 | Train Loss: 0.038694 | Test Loss: 0.032571
Epoch 200/200 | Train Loss: 0.029625 | Test Loss: 0.026158

Training degree 6...
Training 200 epochs with batch size 10
Initial | Train Loss: 25.401697 | Test Loss: 23.695139
Epoch  1/200 | Train Loss: 22.500061 | Test Loss: 17.803903
Epoch 50/200 | Train Loss: 0.051447 | Test Loss: 0.041162
Epoch 100/200 | Train Loss: 0.023288 | Test Loss: 0.018360
Epoch 150/200 | Train Loss: 0.019568 | Test Loss: 0.016984
Epoch 200/200 | Train Loss: 0.017606 | Test Loss: 0.016408

Training degree 7...
Training 200 epochs with batch size 10
Initial | Train Loss: 29.537950 | Test Loss: 39.033805
Epoch  1/200 | Train Loss: 24.809373 | Test Loss: 26.014212
Epoch 50/200 | Train Loss: 0.077886 | Test Loss: 0.200146
Epoch 100/200 | Train Loss: 0.056074 | Test Loss: 0.172809
Epoch 150/200 | Train Loss: 0.043568 | Test Loss: 0.131449
Epoch 200/200 | Train Loss: 0.034304 | Test Loss: 0.099929

Training degree 8...
Training 200 epochs with batch size 10
Initial | Train Loss: 23.632377 | Test Loss: 20.480299
Epoch  1/200 | Train Loss: 20.370792 | Test Loss: 14.205957
Epoch 50/200 | Train Loss: 0.084617 | Test Loss: 0.049612
Epoch 100/200 | Train Loss: 0.047858 | Test Loss: 0.026016
Epoch 150/200 | Train Loss: 0.033757 | Test Loss: 0.018571
Epoch 200/200 | Train Loss: 0.025488 | Test Loss: 0.014358

Training degree 9...
Training 200 epochs with batch size 10
Initial | Train Loss: 35.685659 | Test Loss: 26.614071
Epoch  1/200 | Train Loss: 31.486386 | Test Loss: 16.780702
Epoch 50/200 | Train Loss: 0.163391 | Test Loss: 0.153387
Epoch 100/200 | Train Loss: 0.072829 | Test Loss: 0.072509
Epoch 150/200 | Train Loss: 0.044192 | Test Loss: 0.046898
Epoch 200/200 | Train Loss: 0.029111 | Test Loss: 0.032136

Training degree 10...
Training 200 epochs with batch size 10
Initial | Train Loss: 25.795465 | Test Loss: 25.601852
Epoch  1/200 | Train Loss: 22.238187 | Test Loss: 18.189257
Epoch 50/200 | Train Loss: 0.049879 | Test Loss: 0.070521
Epoch 100/200 | Train Loss: 0.026118 | Test Loss: 0.042324
Epoch 150/200 | Train Loss: 0.021041 | Test Loss: 0.040110
Epoch 200/200 | Train Loss: 0.018661 | Test Loss: 0.038826

Training degree 11...
Training 200 epochs with batch size 10
Initial | Train Loss: 24.626010 | Test Loss: 24.073981
Epoch  1/200 | Train Loss: 21.109928 | Test Loss: 17.602083
Epoch 50/200 | Train Loss: 0.175190 | Test Loss: 0.143591
Epoch 100/200 | Train Loss: 0.025061 | Test Loss: 0.021285
Epoch 150/200 | Train Loss: 0.010525 | Test Loss: 0.016801
Epoch 200/200 | Train Loss: 0.009077 | Test Loss: 0.018243

Training degree 12...
Training 200 epochs with batch size 10
Initial | Train Loss: 21.691311 | Test Loss: 24.945894
Epoch  1/200 | Train Loss: 18.768637 | Test Loss: 18.630286
Epoch 50/200 | Train Loss: 0.103992 | Test Loss: 0.418581
Epoch 100/200 | Train Loss: 0.035578 | Test Loss: 0.340914
Epoch 150/200 | Train Loss: 0.026337 | Test Loss: 0.220440
Epoch 200/200 | Train Loss: 0.022580 | Test Loss: 0.150441

Training degree 13...
Training 200 epochs with batch size 10
Initial | Train Loss: 21.438460 | Test Loss: 19.993141
Epoch  1/200 | Train Loss: 19.023711 | Test Loss: 14.434620
Epoch 50/200 | Train Loss: 0.100873 | Test Loss: 0.105909
Epoch 100/200 | Train Loss: 0.039608 | Test Loss: 0.052066
Epoch 150/200 | Train Loss: 0.027339 | Test Loss: 0.031510
Epoch 200/200 | Train Loss: 0.023127 | Test Loss: 0.023392

Training degree 14...
Training 200 epochs with batch size 10
Initial | Train Loss: 25.055032 | Test Loss: 22.600666
Epoch  1/200 | Train Loss: 22.362081 | Test Loss: 16.011020
Epoch 50/200 | Train Loss: 0.044040 | Test Loss: 0.115732
Epoch 100/200 | Train Loss: 0.033455 | Test Loss: 0.128355
Epoch 150/200 | Train Loss: 0.029995 | Test Loss: 0.116857
Epoch 200/200 | Train Loss: 0.027027 | Test Loss: 0.103254

Training degree 15...
Training 200 epochs with batch size 10
Initial | Train Loss: 21.792075 | Test Loss: 40.881503
Epoch  1/200 | Train Loss: 19.223385 | Test Loss: 32.017646
Epoch 50/200 | Train Loss: 0.067787 | Test Loss: 0.191998
Epoch 100/200 | Train Loss: 0.015577 | Test Loss: 0.449722
Epoch 150/200 | Train Loss: 0.014653 | Test Loss: 0.443645
Epoch 200/200 | Train Loss: 0.014106 | Test Loss: 0.410140

Training degree 16...
Training 200 epochs with batch size 10
Initial | Train Loss: 24.575572 | Test Loss: 24.118043
Epoch  1/200 | Train Loss: 21.415779 | Test Loss: 16.825496
Epoch 50/200 | Train Loss: 0.075685 | Test Loss: 0.102868
Epoch 100/200 | Train Loss: 0.033104 | Test Loss: 0.049790
Epoch 150/200 | Train Loss: 0.024722 | Test Loss: 0.033249
Epoch 200/200 | Train Loss: 0.021098 | Test Loss: 0.027192

Training degree 17...
Training 200 epochs with batch size 10
Initial | Train Loss: 21.077775 | Test Loss: 25.823883
Epoch  1/200 | Train Loss: 18.325922 | Test Loss: 20.168945
Epoch 50/200 | Train Loss: 0.127028 | Test Loss: 0.171675
Epoch 100/200 | Train Loss: 0.040002 | Test Loss: 0.041395
Epoch 150/200 | Train Loss: 0.023549 | Test Loss: 0.046670
Epoch 200/200 | Train Loss: 0.018535 | Test Loss: 0.054757

Training degree 18...
Training 200 epochs with batch size 10
Initial | Train Loss: 20.341548 | Test Loss: 33.302814
Epoch  1/200 | Train Loss: 17.552386 | Test Loss: 27.251511
Epoch 50/200 | Train Loss: 0.137347 | Test Loss: 0.333460
Epoch 100/200 | Train Loss: 0.042638 | Test Loss: 0.069515
Epoch 150/200 | Train Loss: 0.021462 | Test Loss: 0.081360
Epoch 200/200 | Train Loss: 0.016022 | Test Loss: 0.106515

Training degree 19...
Training 200 epochs with batch size 10
Initial | Train Loss: 21.409490 | Test Loss: 23.211001
Epoch  1/200 | Train Loss: 18.940963 | Test Loss: 17.236158
Epoch 50/200 | Train Loss: 0.119734 | Test Loss: 0.136647
Epoch 100/200 | Train Loss: 0.028774 | Test Loss: 0.050776
Epoch 150/200 | Train Loss: 0.015347 | Test Loss: 0.046907
Epoch 200/200 | Train Loss: 0.012761 | Test Loss: 0.046216

Training degree 20...
Training 200 epochs with batch size 10
Initial | Train Loss: 20.992237 | Test Loss: 20.450381
Epoch  1/200 | Train Loss: 18.406041 | Test Loss: 14.677448
Epoch 50/200 | Train Loss: 0.103559 | Test Loss: 0.207371
Epoch 100/200 | Train Loss: 0.039745 | Test Loss: 0.106191
Epoch 150/200 | Train Loss: 0.023977 | Test Loss: 0.055417
Epoch 200/200 | Train Loss: 0.018524 | Test Loss: 0.034386

Training with 10 training examples...
Training 400 epochs with batch size 10
Initial | Train Loss: 22.466125 | Test Loss: 31.433409
Epoch  1/400 | Train Loss: 22.466124 | Test Loss: 30.543976
Epoch 50/400 | Train Loss: 6.970229 | Test Loss: 9.220340
Epoch 100/400 | Train Loss: 2.774495 | Test Loss: 3.982187
Epoch 150/400 | Train Loss: 1.363164 | Test Loss: 2.374231
Epoch 200/400 | Train Loss: 0.785705 | Test Loss: 1.628327
Epoch 250/400 | Train Loss: 0.496006 | Test Loss: 1.148105
Epoch 300/400 | Train Loss: 0.327274 | Test Loss: 0.804412
Epoch 350/400 | Train Loss: 0.220620 | Test Loss: 0.558028
Epoch 400/400 | Train Loss: 0.150609 | Test Loss: 0.385163

Training with 20 training examples...
Training 400 epochs with batch size 10
Initial | Train Loss: 15.819860 | Test Loss: 16.207982
Epoch  1/400 | Train Loss: 15.577960 | Test Loss: 15.203385
Epoch 50/400 | Train Loss: 2.492617 | Test Loss: 3.082035
Epoch 100/400 | Train Loss: 1.312521 | Test Loss: 1.677624
Epoch 150/400 | Train Loss: 0.768918 | Test Loss: 0.924218
Epoch 200/400 | Train Loss: 0.458179 | Test Loss: 0.509180
Epoch 250/400 | Train Loss: 0.275415 | Test Loss: 0.291301
Epoch 300/400 | Train Loss: 0.168581 | Test Loss: 0.182777
Epoch 350/400 | Train Loss: 0.106636 | Test Loss: 0.132922
Epoch 400/400 | Train Loss: 0.070040 | Test Loss: 0.113243

Training with 50 training examples...
Training 400 epochs with batch size 10
Initial | Train Loss: 26.016374 | Test Loss: 24.713722
Epoch  1/400 | Train Loss: 24.116938 | Test Loss: 20.317389
Epoch 50/400 | Train Loss: 0.569712 | Test Loss: 0.506842
Epoch 100/400 | Train Loss: 0.101314 | Test Loss: 0.098375
Epoch 150/400 | Train Loss: 0.041498 | Test Loss: 0.044836
Epoch 200/400 | Train Loss: 0.021189 | Test Loss: 0.025583
Epoch 250/400 | Train Loss: 0.013275 | Test Loss: 0.017462
Epoch 300/400 | Train Loss: 0.010148 | Test Loss: 0.013889
Epoch 350/400 | Train Loss: 0.008930 | Test Loss: 0.012275
Epoch 400/400 | Train Loss: 0.008428 | Test Loss: 0.011529

Training with 100 training examples...
Training 400 epochs with batch size 10
Initial | Train Loss: 24.543432 | Test Loss: 30.333171
Epoch  1/400 | Train Loss: 21.661638 | Test Loss: 22.167819
Epoch 50/400 | Train Loss: 0.064367 | Test Loss: 0.095708
Epoch 100/400 | Train Loss: 0.015942 | Test Loss: 0.022848
Epoch 150/400 | Train Loss: 0.012310 | Test Loss: 0.014583
Epoch 200/400 | Train Loss: 0.010781 | Test Loss: 0.011295
Epoch 250/400 | Train Loss: 0.010038 | Test Loss: 0.010730
Epoch 300/400 | Train Loss: 0.009711 | Test Loss: 0.011188
Epoch 350/400 | Train Loss: 0.009550 | Test Loss: 0.011883
Epoch 400/400 | Train Loss: 0.009459 | Test Loss: 0.012555

Training with 200 training examples...
Training 400 epochs with batch size 10
Initial | Train Loss: 20.984422 | Test Loss: 21.199486
Epoch  1/400 | Train Loss: 15.783926 | Test Loss: 11.499205
Epoch 50/400 | Train Loss: 0.016238 | Test Loss: 0.011523
Epoch 100/400 | Train Loss: 0.010680 | Test Loss: 0.007252
Epoch 150/400 | Train Loss: 0.010647 | Test Loss: 0.007299
Epoch 200/400 | Train Loss: 0.010642 | Test Loss: 0.007317
Epoch 250/400 | Train Loss: 0.010643 | Test Loss: 0.007259
Epoch 300/400 | Train Loss: 0.010659 | Test Loss: 0.007281
Epoch 350/400 | Train Loss: 0.010664 | Test Loss: 0.007252
Epoch 400/400 | Train Loss: 0.010650 | Test Loss: 0.007288

Training with 500 training examples...
Training 400 epochs with batch size 10
Initial | Train Loss: 22.240944 | Test Loss: 22.786739
Epoch  1/400 | Train Loss: 12.881457 | Test Loss: 7.370230
Epoch 50/400 | Train Loss: 0.009416 | Test Loss: 0.010190
Epoch 100/400 | Train Loss: 0.009429 | Test Loss: 0.010143
Epoch 150/400 | Train Loss: 0.009423 | Test Loss: 0.010149
Epoch 200/400 | Train Loss: 0.009429 | Test Loss: 0.010154
Epoch 250/400 | Train Loss: 0.009418 | Test Loss: 0.010149
Epoch 300/400 | Train Loss: 0.009416 | Test Loss: 0.010194
Epoch 350/400 | Train Loss: 0.009424 | Test Loss: 0.010153
Epoch 400/400 | Train Loss: 0.009415 | Test Loss: 0.010144

```

### 3. 如果不对多项式特征x i 进行标准化(1/i!)，会发生什么事情？能用其他方法解决这个问题吗？

### 4. 泛化误差可能为零吗

理论上可能，实际中不可能：

- 必要条件：
模型复杂度 ≥ 真实数据生成过程复杂度
无限训练数据（覆盖输入分布）
无观测噪声（$\epsilon=0$）
- 实际限制：
噪声不可避免
有限数据导致估计偏差
计算精度限制
- proof:
 泛化误差 $\mathcal{E}(f) = \mathbb{E}_{x,y}[(f(x)-y)^2] \geq \mathrm{Var}(\epsilon)$
当 $\epsilon \sim \mathcal{N}(0,\sigma^2)$ 时，$\min_f \mathcal{E}(f) = \sigma^2 > 0$

> 当然还有一种情况，你训练数据集泄露了~

---

## 4.5 权重衰减

权重衰减（weight decay）是一种正则化方法，通过在损失函数中添加一个权重衰减项，使得模型参数的范数（权重向量的模）不断减小，从而限制模型的复杂度。

### L2 正则化

L2 正则化（L2 regularization）是一种权重衰减方法，其正则化项为权重向量的 L2 范数的平方。

$$
\text{Loss}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N L(f(x_i; \mathbf{w}), y_i) + \frac{\lambda}{2} \|\mathbf{w}\|_2^2
$$

关于L2正则化的引入，需要注意的是，低维简单的模型本身并不容易过拟合，因而在此类模型中引入正则化的效果并不明显。而在高维复杂模型中，正则化的效果尤为明显。

二维的模型结果可见一斑：

```text
最终参数对比:
无正则化模型: weight = 3.0044, bias = 2.0535
带L2正则化模型: weight = 2.9628, bias = 1.6508
真实参数: weight = 3.0, bias = 2.0
```

而在全特征:真实特征 100:5的模型中，权重衰减的效果更加明显：

```text
Metric           No Regularization With Regularization
Training MSE              1.2991       1.3801
Test MSE                3.7008       3.3347
Overfitting Gap            2.4017       1.9547
Avg |Noise Weight|           0.1166       0.1071

Effective Feature MAE:         0.1172       0.1636
```

观察到训练误差增大，但是测试误差减小，说明模型的泛化能力更强

### 1. 在本节的估计问题中使用λ的值进行实验。绘制训练和测试精度关于λ的函数。观察到了什么？

$\lambda$ 过小则正则化作用不明显，过大则会导致模型欠拟合。

### 2. 使用验证集来找到最佳值λ。它真的是最优值吗？这有关系吗？

- 如何操作： 我们将数据集分为训练集、验证集和测试集。在训练集上用不同的 λ 值训练多个模型，然后在验证集上评估这些模型的性能（计算精度或损失）。选择在验证集上性能最好的那个模型所对应的 λ 值，作为我们的“最佳值”。
- 它真的是最优值吗？ 不完全是，但没啥关系。
 不是绝对最优的原因： 这个“最佳值”是在我们拥有的特定验证集上找到的。验证集本身只是从整体数据分布中抽取的一个样本，它可能带有一定的随机性（噪音）。如果我们换一个不同的验证集，找到的“最佳λ”可能会略有不同。
- 为什么没关系： 机器学习的实践在很大程度上是基于经验估计和统计意义的。我们不需要一个数学上绝对完美的 λ，我们只需要一个能让我们模型在未见过的数据（测试集）上表现足够好、泛化能力强的值。通过使用验证集，我们正是为了对未知数据（测试集）的性能做一个无偏的估计。只要验证集是 representative（有代表性）的，这个方法就非常有效且实用。

### 3. 如果我们使用 P i |w i |作为我们选择的惩罚（L 1 正则化），那么更新方程会是什么样子？

首先，损失函数变为：
$J(\mathbf{w}) = \text{Loss}(\mathbf{w}) + \lambda \|\mathbf{w}\|_1 = \text{Loss}(\mathbf{w}) + \lambda \sum_i |w_i|$
我们对权重 $w_j$ 求梯度（注意：L1 范数在零点不可导，但在深度学习框架中通常使用次梯度处理）：
$\frac{\partial J}{\partial w_j} = \frac{\partial \text{Loss}}{\partial w_j} + \lambda \cdot \text{sign}(w_j)$
其中，$\text{sign}()$ 是符号函数，即：
$\text{sign}(w_j) = \begin{cases} +1 & \text{if } w_j > 0 \\ -1 & \text{if } w_j < 0 \\ 0 & \text{if } w_j = 0 \end{cases}$
因此，使用梯度下降的权重更新方程为：
$w_j \leftarrow w_j - \eta \frac{\partial J}{\partial w_j} = w_j - \eta \left( \frac{\partial \text{Loss}}{\partial w_j} \right) - \eta\lambda \cdot \text{sign}(w_j)$

L1 正则化的一大特点是能产生稀疏解（许多权重恰好为0），相当于一种自动的特征选择。而 L2 正则化则产生分散的小权重。根据你的需求（是想要稀疏模型还是只是防止过拟合），可以选择不同的正则化方法。

### 4. 我们知道∥w∥ 2 = w ⊤ w。能找到类似的矩阵方程吗（Hint:Frobenius范数）？

Frobenius 范数是衡量矩阵大小的一种常用方法。对于矩阵 $\mathbf{W} \in \mathbb{R}^{m \times n}$，其 Frobenius 范数的平方定义为所有矩阵元素的平方和：
$\|\mathbf{W}\|_F^2 = \sum_{i=1}^{m} \sum_{j=1}^{n} |w_{ij}|^2$
这与向量的 L2 范数平方（wᵀw）的形式是完全一致的，只是扩展到了矩阵。它同样可以写成矩阵迹运算的形式：
$\|\mathbf{W}\|_F^2 = \text{tr}(\mathbf{W}^\top \mathbf{W})$
因为矩阵 $\mathbf{W}^\top \mathbf{W}$ 的对角线元素正好是 $\mathbf{W}$ 的每一列向量的 L2 范数平方。

在深度学习中，当我们需要对整个权重矩阵（而不仅仅是一个权重向量）进行正则化时，就会使用 Frobenius 范数。例如，在全连接层或卷积层中，权重衰减项通常是 $\frac{\lambda}{2} \|\mathbf{W}\|_F^2$。

### 5. 回顾训练误差和泛化误差之间的关系。除了权重衰减、增加训练数据、使用适当复杂度的模型之外，还能想出其他什么方法来处理过拟合

过拟合的本质是模型学习了训练数据中的噪音和非普遍特征。除了你提到的三种经典方法，还有非常多且有效的手段：

- Dropout： 在训练过程中，随机地“丢弃”（暂时移除）神经网络中的一部分神经元。这可以防止神经元之间产生复杂的协同适应，迫使每个神经元都能独立发挥重要作用，从而增强模型的鲁棒性和泛化能力。这是一种非常强大且常用的正则化技术。
- 早停： 在训练过程中，持续监控验证集上的性能。当验证集上的损失不再下降反而开始上升时（即模型开始过拟合），就立即停止训练。
- 数据增强： 主要用于图像、语音等领域。通过对训练数据进行一系列随机但合理的变换（如图像的旋转、裁剪、颜色抖动等），可以人工扩大训练数据集的大小和多样性，让模型看到更多可能的变化，从而学习到更泛化的特征。
- 批归一化： 虽然其主要目的是稳定和加速训练过程，但因为它为每一层的输入引入了轻微的随机噪音（来自于mini-batch的统计估计），所以它也具有一定的正则化效果，可以减少甚至替代Dropout。
- 标签平滑： 一种正则化技术，通过软化硬标签（例如，将 [1, 0, 0] 变为 [0.9, 0.05, 0.05]）来防止模型对标签过于自信，从而减轻过拟合。
- 模型集成： 训练多个不同的模型，然后用它们的平均或投票结果作为最终预测。因为不同模型通常不会在相同的点上过拟合，集成可以平均掉过拟合的噪音，获得更好的泛化性能。
- 添加噪音： 在输入数据或网络的隐藏层激活值上添加微小的随机噪音，同样可以迫使模型变得更强健。

---

## 4.6 Dropout

Dropout 是一种正则化技术，它在训练过程中随机丢弃一些神经元来防止过拟合。

### 1. 如果更改第一层和第二层的暂退法概率，会发生什么情况？具体地说，如果交换这两个层，会发生什么情况？设计一个实验来回答这些问题，定量描述该结果，并总结定性的结论

差别：

```text
--- original model ---
使用设备: cuda
Epoch [1/10], Train Loss: 0.5594, Train Acc: 79.89%, Test Loss: 0.4238, Test Acc: 84.53%
Epoch [2/10], Train Loss: 0.3712, Train Acc: 86.25%, Test Loss: 0.3907, Test Acc: 85.84%
Epoch [3/10], Train Loss: 0.3295, Train Acc: 87.82%, Test Loss: 0.3864, Test Acc: 85.88%
Epoch [4/10], Train Loss: 0.3058, Train Acc: 88.65%, Test Loss: 0.3728, Test Acc: 86.57%
Epoch [5/10], Train Loss: 0.2854, Train Acc: 89.34%, Test Loss: 0.3612, Test Acc: 87.00%
Epoch [6/10], Train Loss: 0.2651, Train Acc: 89.98%, Test Loss: 0.3620, Test Acc: 87.13%
Epoch [7/10], Train Loss: 0.2519, Train Acc: 90.42%, Test Loss: 0.3591, Test Acc: 87.35%
Epoch [8/10], Train Loss: 0.2348, Train Acc: 90.94%, Test Loss: 0.3432, Test Acc: 88.18%
Epoch [9/10], Train Loss: 0.2248, Train Acc: 91.42%, Test Loss: 0.3375, Test Acc: 88.58%
Epoch [10/10], Train Loss: 0.2146, Train Acc: 91.83%, Test Loss: 0.3393, Test Acc: 88.77%
训练时间: 86.85秒
--- flip dropout ---
使用设备: cuda
Epoch [1/10], Train Loss: 0.5789, Train Acc: 78.89%, Test Loss: 0.4756, Test Acc: 82.79%
Epoch [2/10], Train Loss: 0.3827, Train Acc: 85.79%, Test Loss: 0.3919, Test Acc: 85.91%
Epoch [3/10], Train Loss: 0.3376, Train Acc: 87.45%, Test Loss: 0.3802, Test Acc: 85.91%
Epoch [4/10], Train Loss: 0.3088, Train Acc: 88.56%, Test Loss: 0.3717, Test Acc: 86.89%
Epoch [5/10], Train Loss: 0.2850, Train Acc: 89.37%, Test Loss: 0.3833, Test Acc: 86.62%
Epoch [6/10], Train Loss: 0.2693, Train Acc: 89.86%, Test Loss: 0.3865, Test Acc: 86.80%
Epoch [7/10], Train Loss: 0.2519, Train Acc: 90.52%, Test Loss: 0.3331, Test Acc: 88.20%
Epoch [8/10], Train Loss: 0.2387, Train Acc: 90.98%, Test Loss: 0.3347, Test Acc: 88.42%
Epoch [9/10], Train Loss: 0.2247, Train Acc: 91.49%, Test Loss: 0.3732, Test Acc: 87.41%
Epoch [10/10], Train Loss: 0.2126, Train Acc: 91.93%, Test Loss: 0.3515, Test Acc: 88.30%
训练时间: 90.83秒
```

分析：

- 原训练（dropout = [0.2, 0.5]）：第一层（接近输入层）使用较低的dropout率（0.2），保留了更多输入信息，使模型能更好地学习基础特征；第二层（接近输出层）使用较高的dropout率（0.5），增强了正则化效果，防止过拟合。训练过程稳定，测试损失和准确率平滑改善。
- flip：第一层高dropout率（0.5）会随机丢弃50%的输入特征，导致模型难以学习稳定的基础特征，引入大量噪声；第二层低dropout率（0.2）则正则化不足，容易使模型过拟合训练数据。这种不平衡的正则化导致优化过程不稳定，从而引起测试性能的波动
- 第一层高dropout率会使梯度计算变得嘈杂（noisy），因为每次迭代时输入特征都被大量丢弃。这可能导致梯度下降方向不一致，使训练损失和测试损失出现震荡。
- 从第二次训练结果看，测试损失在epoch 5-6上升（0.3833 → 0.3865），然后在epoch 7下降（0.3331），之后又上升（epoch 9: 0.3732），这种波动表明模型在优化过程中没有收敛到稳定点，而是陷入了局部极小值或 saddle point。
- 第二层低dropout率（0.2）不足以防止过拟合。在训练后期，模型可能开始过拟合训练数据（训练准确率高达91.93%），但测试准确率却波动（88.30%），这表明模型对测试集的泛化能力不稳定。
- 第一层高dropout率又可能导致欠拟合，因为输入信息被过度丢弃，模型无法充分学习特征。这种矛盾使得模型在过拟合和欠拟合之间摇摆，导致测试性能波动。
- Dropout本身具有随机性，但第二次设置的随机性更大（第一层50%丢弃），这放大了训练过程中的方差。即使使用相同的随机种子，高dropout率也会导致每次迭代的网络结构差异更大，从而增加波动。

### 2. 增加训练轮数，并将使用暂退法和不使用暂退法时获得的结果进行比较

- 使用Dropout的模型理论上泛化能力更好
- 无Dropout的模型可能训练准确率更高但测试准确率较低，存在过拟合

### 3. 当应用或不应用暂退法时，每个隐藏层中激活值的方差是多少？绘制一个曲线图，以显示这两个模型的每个隐藏层中激活值的方差是如何随时间变化的

- 从方差曲线图可以看出，使用Dropout的模型激活值方差更稳定
- 无Dropout的模型激活值方差随着训练进行而增大，表明激活值分布变化较大

### 4. 为什么在测试时通常不使用暂退法？

- Dropout在训练时是一种集成学习的方法，测试时需要'平均'所有子模型的结果
- 测试时我们希望使用完整的模型能力进行预测，而不是随机丢弃部分神经元
- 在实现上，测试时Dropout层会被关闭，但激活值会按Dropout比例缩放以保持期望值一致

### 5. 以本节中的模型为例，比较使用暂退法和权重衰减的效果。如果同时使用暂退法和权重衰减，会发生什么情况？结果是累加的吗？收益是否减少（或者说更糟）？它们互相抵消了吗？

- Dropout和权重衰减都能提高泛化性能，但机制不同
- Dropout通过随机丢弃神经元防止过拟合，权重衰减通过惩罚大权重值
- 同时使用两者时，效果通常是累加的，但可能不是简单的线性叠加
- 此外过度正则化可能导致欠拟合，需要仔细调参

### 6. 如果我们将暂退法应用到权重矩阵的各个权重，而不是激活值，会发生什么？

- 该技术称之为DropConnect，与Dropout类似但丢弃的是权重而不是激活值
- DropConnect可能提供不同的正则化效果，但实现更复杂
- 在某些任务上可能表现更好，但计算开销更大

### 7. 发明另一种用于在每一层注入随机噪声的技术，该技术不同于标准的暂退法技术。尝试开发一种在Fashion‐MNIST数据集（对于固定架构）上性能优于暂退法的方法

- 代码实现的是高斯噪声注入，即在输入层添加高斯噪声
- 高斯噪声的参数(标准差)需要仔细调整，过大或过小都会影响性能
- 其他技术还包括Layer噪声、梯度噪声等

---

## 4.7 前向传播和反向传播

### 什么是前向传播？

将输入数据从网络的第一层（输入层）传递到最后一层（输出层），并最终计算出一个预测值（或一组预测值）。同时，它会计算当前预测与真实值之间的差距，这个差距我们称之为损失 (Loss)。

### 什么是反向传播？

利用微积分中的链式法则，将最终计算出的损失值，从网络的输出层反向传播回第一层，计算出损失函数对于网络中每一个参数（每一个 w 和 b）的梯度。梯度指明了各个参数应该调整的方向和幅度，并最终使得损失函数最小化。

### 过程

#### 前向传播

1. 输入：将一批训练数据（如图像、文本特征）输入到网络的第一层。
2. 线性变换：每一层的每个神经元都会对上一层的输出进行一个线性加权求和操作：z = w * x + b（其中 w 是权重，x 是输入，b 是偏置）。
3. 激活函数：将线性变换的结果 z 输入到一个非线性激活函数（如 ReLU, Sigmoid）中：a = activation_function(z)。这一步至关重要，它为网络引入了非线性，使得神经网络可以拟合极其复杂的函数。
4. 传递：将本层的输出 a 作为下一层的输入。
5. 重复：重复步骤 2-4，直到数据到达输出层。
6. 计算损失：将网络的最终输出与数据的真实标签进行比较，通过一个损失函数 (Loss Function)（如均方误差 MSE、交叉熵 Cross-Entropy）计算出一个标量值，这就是损失。损失值衡量了当前网络预测的“糟糕”程度。
 
#### 反向传播

1. 计算输出层梯度：首先计算损失函数对输出层输出的梯度。
2. 链式反向传播：从输出层开始，反向逐层计算。
  - 计算损失对本层线性输出 z 的梯度。
  - 利用这个梯度，计算损失对本层权重 w 和偏置 b 的梯度。
  - 继续计算损失对上一层激活输出 a 的梯度，并将这个梯度作为上一层的“输入误差”，继续反向传播。
3. 重复：重复步骤 2，直到传播到第一层。
至此，我们得到了一个“梯度列表”，里面包含了损失函数对每一个参数的偏导数（即梯度）。

### Exercises - 4.7

给出模型为：带权重衰减（L2正则化）的单隐藏层多层感知机

1. 假设一些标量函数X的输入X是n × m矩阵。f相对于X的梯度维数是多少？
 标量函数 $f(X)$ 的输出是一个标量，而输入 $X$ 是一个 $n \times m$ 矩阵。梯度 $\nabla_X f$ 是 $f$ 对 $X$ 中每个元素的偏导数组成的矩阵，因此其维数与 $X$ 相同，即 $n \times m$。每个元素 $\frac{\partial f}{\partial X_{ij}}$ 表示 $f$ 对 $X$ 在位置 $(i,j)$ 的变化率。
2. 向模型的隐藏层添加偏置项（不需要在正则化中包含偏置项）。
  1. 画出相应的计算图。
  ![前向计算图](/Multilayer%20Perceptron/Solutions/Forward_Calculate_Graph.png)
  2. 推导正向和反向传播方程。
  正向传播方程:
   输入: $X$ (批量大小 $b \times d$)
   隐藏层线性变换: $Z = X W + b$ (广播 $b$ 到 $b \times h$)
   激活: $A = \text{ReLU}(Z)$ (逐元素)
   输出层线性变换: $O = A V + c$ (广播 $c$ 到 $b \times c$)
   输出激活: $Y_{\text{hat}} = \text{softmax}(O)$ (逐行)
   损失: $L = -\frac{1}{b} \sum_{i=1}^b \sum_{j=1}^c Y_{ij} \log(Y_{\text{hat}_{ij}})$ (交叉熵损失，平均 over batch)
   正则化项: $R = \frac{\lambda}{2} (\|W\|_F^2 + \|V\|_F^2)$
   总损失: $J = L + R$
  反向传播方程:
   首先，计算总损失 $J$ 对输出 $O$ 的梯度。在 softmax 交叉熵下，有：
   $$\frac{\partial L}{\partial O} = Y_{\text{hat}} - Y \quad (\text{维度 } b \times c)$$
   由于 $R$ 不依赖于 $O$，所以 $\frac{\partial J}{\partial O} = \frac{\partial L}{\partial O}$.
   然后，反向传播通过输出层：
   对 $V$ 的梯度:
   $$\frac{\partial J}{\partial V} = \frac{\partial J}{\partial O} \cdot \frac{\partial O}{\partial V} = A^T \frac{\partial J}{\partial O} + \lambda V \quad (\text{维度 } h \times c)$$
   其中 $\frac{\partial O}{\partial V} = A$，且 $R$ 对 $V$ 的梯度为 $\lambda V$.
   对 $c$ 的梯度:
   $$\frac{\partial J}{\partial c} = \sum_{i=1}^b \frac{\partial J}{\partial O_i} \quad (\text{维度 } 1 \times c)$$
   其中 $\frac{\partial O}{\partial c} = 1$，且 $R$ 不依赖于 $c$.
   接下来，计算对 $A$ 的梯度:
   $$\frac{\partial J}{\partial A} = \frac{\partial J}{\partial O} \cdot \frac{\partial O}{\partial A} = \frac{\partial J}{\partial O} V^T \quad (\text{维度 } b \times h)$$
   其中 $\frac{\partial O}{\partial A} = V$.
   通过激活层 (ReLU):
   $$\frac{\partial J}{\partial Z} = \frac{\partial J}{\partial A} \odot \text{ReLU}'(Z) \quad (\text{维度 } b \times h)$$
   其中 $\text{ReLU}'(Z)$ 是逐元素的导数，$Z > 0$ 时为 1，否则为 0。
   最后，通过隐藏层线性变换:
   对 $W$ 的梯度:
   $$\frac{\partial J}{\partial W} = \frac{\partial J}{\partial Z} \cdot \frac{\partial Z}{\partial W} = X^T \frac{\partial J}{\partial Z} + \lambda W \quad (\text{维度 } d \times h)$$
   其中 $\frac{\partial Z}{\partial W} = X$，且 $R$ 对 $W$ 的梯度为 $\lambda W$.
   对 $b$ 的梯度:
   $$\frac{\partial J}{\partial b} = \sum_{i=1}^b \frac{\partial J}{\partial Z_i} \quad (\text{维度 } 1 \times h)$$
   其中 $\frac{\partial Z}{\partial b} = 1$，且 $R$ 不依赖于 $b$.

3. 计算模型用于训练和预测的内存占用
 内存占用取决于参数存储和激活值存储。在使用 32 位浮点数（4字节）的情况下：
 参数内存:
  $W$: $d \times h$ 参数
  $b$: $h$ 参数
  $V$: $h \times c$ 参数
  $c$: $c$ 参数
 总参数数量: $P = d h + h + h c + c$
 参数内存: $4P$ 字节

 训练内存（前向和反向传播）:
 除了参数，还需要存储中间激活值用于反向传播。关键激活值:
  $Z$: $b \times h$ 元素
  $A$: $b \times h$ 元素
  $O$: $b \times c$ 元素
 $Y_{\text{hat}}$: $b \times c$ 元素（但通常与 $O$ 共享或可推导，但为安全起见计入）
 此外，梯度存储需要与参数相同大小的内存，但通常与参数共享或临时分配。

 激活值内存: $4 \times (b h + b h + b c + b c) = 4b(2h + 2c)$ 字节
 梯度内存: $4P$ 字节（用于 $\nabla W, \nabla b, \nabla V, \nabla c$）
4. 计算本节所描述的模型，用于训练和预测的内存占用。
 略
5. 假设想计算二阶导数。计算图发生了什么？预计计算需要多长时间？
 计算二阶导数意味着计算 Hessian 矩阵或二阶梯度。计算图会变得复杂：
  在现有计算图（用于一阶导数）的基础上，需要构建二阶计算图，即计算梯度的梯度。
  自动微分工具（如 PyTorch 的 torch.autograd.grad）会展开计算图，为每个一阶梯度节点添加二阶梯度节点。
  计算图大小会显著增加，因为每个参数的二阶导数涉及所有其他参数。
 计算时间:
  如果参数数量为 $P$，Hessian 矩阵大小为 $P \times P$，计算和存储成本为 $O(P^2)$。
  对于神经网络，$P$ 通常很大（如数千到数百万），因此计算二阶导数非常昂贵，可能比一阶导数慢 $P$ 倍或更多。
  实际中，通常避免直接计算完整 Hessian，而使用近似方法（如 Hessian-vector products），但即使这样，计算时间也可能增加一个数量级以上。
6. 假设计算图对当前拥有的GPU来说太大了。

  1. 请试着把它划分到多个GPU上。
  划分计算图到多个 GPU 上可以通过模型并行或数据并行实现：
  模型并行: 将模型的不同部分放置在不同 GPU 上。例如：
   将权重矩阵 $W$ 按列分割，每个 GPU 负责一部分隐藏神经元。前向传播时，每个 GPU 计算部分 $Z$ 和 $A$，然后通过通信聚合结果（如 all-gather）用于后续层。
  类似地，将 $V$ 按行或列分割。
  数据并行: 将批量数据分割到多个 GPU 上，每个 GPU 有完整的模型副本，处理一个子批量。然后同步梯度（如 all-reduce）来更新模型。这是更常见的方法。
  对于单隐藏层 MLP，数据并行通常更简单有效。模型并行可能引入通信开销，但适用于极大模型。
  2. 与小批量训练相比，有哪些优点和缺点？
  优点:
   处理更大模型: 通过模型并行，可以将大型模型分布到多个 GPU 上，克服单个 GPU 内存限制。
   处理更大批量: 通过数据并行，可以使用更大的有效批量大小，加快训练速度（特别是与梯度累积结合）。
   更快训练: 并行计算减少训练时间，尤其当计算是瓶颈时。
   更好的泛化: 有时使用更大批量可以提高训练稳定性，但可能需要调整学习率。
  缺点:
   通信开销: 在 GPU 之间同步梯度或激活值引入通信延迟，可能成为瓶颈（尤其是低速互联）。
   实现复杂性: 需要处理分布式训练逻辑（如梯度同步、数据加载），增加代码复杂度。
   资源成本: 需要多个 GPU，增加硬件成本。
   收敛问题: 更大批量可能影响收敛特性，需要谨慎调整超参数（如学习率）。

---

## 4.8 数值稳定性和模型初始化

### 4.8.1 梯度消失与爆炸

#### 梯度消失

梯度消失是指在深层网络中，随着网络层数加深，梯度逐渐变小或消失的现象
常见的诱因包括：

- 学习率设置过小，导致更新步长过小
- 激活函数的选择不恰当，如 ReLU
- 网络层数过多

#### 梯度爆炸

梯度爆炸是指在深层网络中，随着网络层数加深，梯度逐渐增大或爆炸的现象

### 4.8.2 参数初始化

Xavier初始化

### Exercises - 4.8

1. 除了多层感知机的排列对称性之外，还能设计出其他神经网络可能会表现出对称性且需要被打破的情况吗？
  当然可以。神经网络的对称性不止存在于MLP中，打破这些对称性是初始化的重要目标。以下是一些其他典型的例子：
     - **卷积神经网络中的平移对称性 (Translation Symmetry):**
       - **现象:** 理论上，如果一个卷积层的所有卷积核参数都被初始化为相同的值，那么无论卷积核在输入图像的哪个位置进行滑动计算，其输出特征图（Feature Map）的每个位置的值都将完全相同。这意味着网络无法捕捉到图像中不同位置的特征（例如，猫的眼睛在左上角和右下角对网络来说是一样的）。
       - **如何打破:** 通过**随机初始化**每个卷积核的权重，赋予每个卷积核独特的“使命”（如检测边缘、颜色块等），从而打破这种对称性，让网络能够学习到位置敏感的特征。
     - **循环神经网络中的时间步对称性 (Timestep Symmetry):**
       - **现象:** 在RNN（如简单RNN）中，如果在每个时间步都使用相同的权重矩阵且初始状态为零，并且输入序列是平稳的，那么网络对每个时间步的处理可能会表现出对称性，难以捕捉序列中的时间动态和长期依赖关系。
       - **如何打破:** 除了随机初始化权重，更关键的是使用**非零的、小幅随机**的初始隐藏状态，或者使用像LSTM、GRU这样具有更复杂门控机制的单元，其内部结构本身就更有利于打破这种对称性。
     - **对称网络结构 (Symmetrical Architectures):**
       - **现象:** 如果你刻意设计一个对称的网络结构，例如一个具有对称分支的Siamese网络（连体网络）或某些特定的图神经网络，其子网络部分在初始化时如果参数相同，则会完全对称。
       - **是否需要打破:** 这取决于任务目标。有时我们**希望保持这种对称性**（例如，Siamese网络的两个分支需要共享参数以确保对称性）。而在其他情况下，我们可能希望分支学习到不同的特征，这时就需要通过不同的初始化来打破对称。
**核心思想：** 打破对称性的根本目的是确保网络中的每个神经元（或一组神经元）在训练初期就能接收到不同的梯度信号，从而能够朝着不同的方向演化，学习到数据中多样化的特征。随机初始化是实现这一目标最直接、最有效的方法。
2. 我们是否可以将线性回归或softmax回归中的所有权重参数初始化为相同的值？
  不能,这是一个非常重要的原则。原因如下：
    - **输出对称性导致梯度对称性：** 以Softmax回归为例，假设你将所有权重 `W` 初始化为同一个值，偏置 `b` 也初始化为同一个值。那么对于任何一个输入样本 `x`，每个输出神经元 `o_i` 的预激活值（`z_i = W_i·x + b_i`）将**完全相同**。
    - **梯度相同，参数更新一致：** 在反向传播计算梯度时，每个权重参数所接收到的梯度信号也会是**完全相同**的（具体推导可以看损失函数对每个 `W_ij` 的偏导）。这意味着在每次参数更新时，所有权重值都会**被更新为完全相同的新值**。
    - **无法学习有效特征：** 整个训练过程中，所有权重始终保持相等。这等价于你只有一个“特征探测器”，而不是多个。模型的能力被极大地限制了，它永远无法学习到输入特征与不同输出类别之间复杂多样的映射关系，最终性能会非常差。
  **结论：** 即使是在最简单的线性模型中，**随机初始化**也是必须的，以确保每个参数都能接收到独特的梯度信号，从而能够独立地学习。

3. 在相关资料中查找两个矩阵乘积特征值的解析界。这对确保梯度条件合适有什么启示？

这个问题涉及到深度学习中梯度稳定性的理论核心。

- **解析界 (Analytical Bounds):**
 对于两个矩阵 `A` 和 `B` 的乘积，其特征值没有像奇异值那样直接而简洁的界。但我们可以利用**奇异值**来间接分析，因为奇异值决定了矩阵变换对向量长度的最大和最小缩放倍数，这与梯度爆炸/消失直接相关。
 一个关键的不等式是：`σ_min(A)σ_min(B) ≤ σ_i(AB) ≤ σ_max(A)σ_max(B)`
 其中 `σ_max(·)` 和 `σ_min(·)` 代表最大和最小奇异值。
- **对梯度条件的启示:**
 在深度神经网络中，反向传播的过程涉及到一连串矩阵乘法（雅可比矩阵）。例如，一个L层的网络，最终损失对第一层权重的梯度可以表示为多个雅可比矩阵的连乘。
 根据上面的奇异值不等式，这个连乘矩阵的奇异值范围大约是所有层雅可比矩阵奇异值范围的**乘积**。
  - **梯度爆炸 (Exploding Gradient):** 如果每一层的 `σ_max > 1`，那么多层连乘后，最大奇异值会指数级增长 (`(σ_max)^L`)，导致梯度变得巨大。
  - **梯度消失 (Vanishing Gradient):** 如果每一层的 `σ_max < 1`（对于Sigmoid/Tanh激活函数，这是常态），那么多层连乘后，最大奇异值会指数级衰减 (`(σ_max)^L`)，导致梯度变得接近零。

- **实践指导意义:**
  1. **初始化的重要性:** 这解释了为什么我们需要如Xavier或He初始化等方法。这些方法的目标正是**确保每一层输出的方差是稳定的**，这等价于在初始化时让权重矩阵的奇异值大约在1附近，从而在训练开始时避免梯度爆炸或消失。
  2. **激活函数的选择:** 这解释了ReLU及其变体（如Leaky ReLU）为什么能缓解梯度消失问题。因为它们在正区间的导数为1，避免了像Sigmoid函数那样将梯度收缩到小于1的范围。
  3. **架构设计:** 这为**残差连接（ResNet）** 和**批量归一化（BatchNorm）** 等技术的有效性提供了理论视角。残差连接创造了一条“高速公路”，让梯度可以直接回传，绕过了可能引发衰减的矩阵连乘路径。BatchNorm通过稳定每层的输入分布，间接地帮助稳定了梯度的传播。

---

### 4. 如果我们知道某些项是发散的，我们能在事后修正吗？

**可以，有一系列成熟的“事后修正”技术来处理训练过程中的发散问题。**

“发散”通常指损失函数（Loss）或梯度（Gradients）的值变得异常大（NaN或Inf），导致训练无法继续进行。常见的修正策略包括：

1. **梯度裁剪 (Gradient Clipping):**
   - **这是最常用、最直接的事后修正方法。** 在反向传播计算出梯度之后，但在参数更新之前，检查梯度向量的范数（通常是L2范数）。如果这个范数超过了一个预设的阈值（clip_value），就将整个梯度向量**按比例缩放**，使其范数等于阈值。
   - `g = min(1, clip_value / ‖g‖) * g`
   - 这相当于给梯度设置了一个上限，能非常有效地防止因梯度爆炸而导致的参数更新步伐过大和训练发散。它在RNN中尤其重要。
2. **降低学习率 (Reducing Learning Rate):**
   - 如果你观察到训练损失在持续下降后突然开始上升并发散（如下图），这通常意味着优化器“冲过了”最低点。最直接的补救措施就是**手动或通过调度器（Scheduler）降低学习率**。
   - `new_lr = old_lr * 0.1` （或0.5等其他因子）
   - 一个更自动化的方法是使用**梯度裁剪** + **学习率 warmup**，在训练初期逐渐增大学习率，这有助于稳定训练。
3. **回滚到检查点 (Rolling Back to Checkpoints):**
   - 这是一个非常实用的工程实践。在训练过程中，定期保存模型的 checkpoint（权重和优化器状态）。一旦发生发散，就**停止当前训练，加载发散前最后一个稳定的checkpoint**，然后采取上述措施（如降低学习率、启用梯度裁剪）重新开始训练。
4. **更改优化器 (Switching Optimizers):**
   - 一些自适应优化器（如 **Adam**）由于其内置的 per-parameter 学习率调整机制，相比简单的SGD，对梯度爆炸的鲁棒性更强。如果你用SGD时遇到发散，可以尝试换用Adam。
   - **注意：** Adam等优化器本身也有超参（如 `epsilon`），设置不当也可能导致数值问题，但通常比SGD更稳定。
