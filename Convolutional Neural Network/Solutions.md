# Chapter 6 卷积神经网络 - Convolutional Neural Networks

## 6.1 从全连接层到卷积

### Exercises 6.1

#### 假设卷积层 (6.1.3)覆盖的局部区域 $\Delta = 0$ 。在这种情况下，证明卷积内核为每组通道独立地实现一个全连接层

在标准卷积神经网络中，卷积操作涉及一个四维卷积核张量 $ V $ 形状为 $[k_h, k_w, C_{\text{in}}, C_{\text{out}}]$，其中：

- $ k_h $ 和 $ k_w $ 是卷积核的空间尺寸（高度和宽度）。
- $ C_{\text{in}} $ 是输入通道数。
- $ C_{\text{out}} $ 是输出通道数。

对于输入张量 $ X $ 形状为 $[H, W, C_{\text{in}}]$，输出张量 $ H $ 形状为 $[H', W', C_{\text{out}}]$，卷积操作定义为：
$$
H(i, j, m) = \sigma \left( \sum_{a=0}^{k_h-1} \sum_{b=0}^{k_w-1} \sum_{c=0}^{C_{\text{in}}-1} V(a, b, c, m) \cdot X(i+a, j+b, c) + b(m) \right)
$$
其中 $ \sigma $ 是激活函数，$ b $ 是偏置向量。

当卷积核大小为1x1时（Δ=0），在这种情况下，卷积核张量 $ V $ 的形状变为 $[1, 1, C_{\text{in}}, C_{\text{out}}]$。这意味着：

- 卷积核不再覆盖空间邻域（因为 $ a $ 和 $ b $ 只能为0）。
- 卷积操作简化为只对通道维度进行线性组合。

因此，输出公式简化为：
$$
H(i, j, m) = \sigma \left( \sum_{c=0}^{C_{\text{in}}-1} V(0, 0, c, m) \cdot X(i, j, c) + b(m) \right)
$$

现在，考虑每个空间位置 $(i, j)$。输入在该位置是一个向量 $ \mathbf{x}_{i,j} \in \mathbb{R}^{C_{\text{in}}} $，其中 $ \mathbf{x}_{i,j} = [X(i,j,0), X(i,j,1), \dots, X(i,j,C_{\text{in}}-1)] $。输出在该位置是一个向量 $ \mathbf{h}_{i,j} \in \mathbb{R}^{C_{\text{out}}} $，其中 $ \mathbf{h}_{i,j} = [H(i,j,0), H(i,j,1), \dots, H(i,j,C_{\text{out}}-1)] $。

定义权重矩阵 $ W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}}} $，其中 $ W(m, c) = V(0, 0, c, m) $，和偏置向量 $ \mathbf{b} \in \mathbb{R}^{C_{\text{out}}} $。则对于每个空间位置 $(i, j)$，有：
$$
\mathbf{h}_{i,j} = \sigma \left( W \mathbf{x}_{i,j} + \mathbf{b} \right)
$$这正好是全连接层的操作：对输入向量进行线性变换并激活。

#### 为什么平移不变性可能也不是好主意呢？

平移不变性可能也不是好主意。因为它不关心物体的绝对位置，会主动丢弃位置信息，但在位置信息存在作用的情况下，就成了明显缺陷

#### 当从图像边界像素获取隐藏表示时，我们需要思考哪些问题？

当从图像边界像素获取隐藏表示时，我们需要考虑以下问题：

- 边界像素的位置信息是否重要？
- 如何处理边界像素的位置信息？
- 信息可能存在部分缺失，如何处理？
- 处理效率如何？
- 当前边界处理方式是否会引入新的、不被我们希望的特征？

#### 描述一个类似的音频卷积层的架构

| 特性 | 图像卷积 (2D) | 音频一维卷积 (1D) | 音频二维卷积 (2D on Spectrogram) |
| :--- | :--- | :--- | :--- |
| **输入维度** | `[H, W, C_in]` | `[T, C_in]` | `[T, F, C_in]` (Time, Frequency, Channels) |
| **卷积核** | `[k_h, k_w, C_in, C_out]` | `[k_t, C_in, C_out]` | `[k_t, k_f, C_in, C_out]` |
| **感受野** | 空间局部区域 (patch) | **时间**局部区域 (snippet) | **时-频**局部区域 (patch) |
| **滑动方向** | 沿**高度**和**宽度** | 沿**时间**轴 | 沿**时间**和**频率**轴 |
| **平移不变性** | **空间**平移不变性 | **时间**平移不变性 | **时-频**平移不变性 |

#### 卷积层也适合于文本数据吗？为什么？

能，但未必最优。
我们可以将文本转换成适合卷积操作的结构，就像图像是像素矩阵

- 每个单词或子词被映射为一个密集向量（词向量），维度为 d
- 一个长度为 L 的句子就可以表示为一个 2D 矩阵，形状为 [L, d]
- 与图像使用 2D 卷积核（在高和宽上滑动）不同，文本卷积通常使用 一维卷积
- 卷积核的宽度等于词向量的维度 d，高度（或称“窗口大小”）为 k。这意味着它一次查看 k 个连续的单词。
  例如，一个 k=3 的卷积核，其形状为 [3, d]。它会在序列方向（即时间轴，从一个词到下一个词）上滑动，每次计算这 3 个单词向量的一个加权组合，从而产生一个新的特征。多个这样的卷积核可以检测句子中不同的局部模式。

通过卷积，我们能够捕捉到文本的局部模式，如语法和语义，但视野受限于卷积层大小的限制。因此，卷积层可能不适合处理长文本，如文章、微博等。
此外，正如前文提及，卷积的过程中往往会导致位置丢失，即它无法区分一个短语是出现在句首还是句尾，而这有时对语义至关重要。
还有池化(见后文)，虽然池化往往用来降维和整合特征。能最大化的提取最显著的特征，但它也丢弃了序列的顺序和精细的时间信息。

#### 证明在 $(f*g)(i,j)=\sum\limits_{n=-\infty}^\infty\sum\limits_{m=-\infty}^{\infty} f(m, n)g(i-m,j-n)$ 中，$f * g = g * f$

假设卷积定义为：
$$
(f * g)(i, j) = \sum_{m=-\infty}^{\infty} \sum_{n=-\infty}^{\infty} f(m, n) \, g(i - m, j - n)
$$
我们需要证明 $(f * g)(i, j) = (g * f)(i, j)$.

首先，考虑 $(g * f)(i, j)$ 的定义：
$$
(g * f)(i, j) = \sum_{m=-\infty}^{\infty} \sum_{n=-\infty}^{\infty} g(m, n) \, f(i - m, j - n)
$$

现在，进行变量替换。令 $k = i - m$ 和 $l = j - n$。则当 $m$ 和 $n$ 取遍所有整数时，$k$ 和 $l$ 也取遍所有整数。同时，有 $m = i - k$ 和 $n = j - l$。

代入上式：
$$
(g * f)(i, j) = \sum_{k=-\infty}^{\infty} \sum_{l=-\infty}^{\infty} g(i - k, j - l) \, f(k, l)
$$

由于求和变量是哑变量，可以重命名 $k$ 和 $l$ 回 $m$ 和 $n$：
$$
(g * f)(i, j) = \sum_{m=-\infty}^{\infty} \sum_{n=-\infty}^{\infty} g(i - m, j - n) \, f(m, n)
$$

比较 $(f * g)(i, j)$：
$$
(f * g)(i, j) = \sum_{m=-\infty}^{\infty} \sum_{n=-\infty}^{\infty} f(m, n) \, g(i - m, j - n)
$$

由于乘法具有交换性，即 $f(m, n) \, g(i - m, j - n) = g(i - m, j - n) \, f(m, n)$，因此两个求和表达式 identical：
$$
(f *g)(i, j) = (g* f)(i, j)
$$

因此，卷积操作是可交换的。

## 6.2 图像卷积

### Exercises 6.2

####  构建一个具有对角线边缘的图像X

1. 如果将本节中举例的卷积核K应用于X，会发生什么情况？
2. 如果转置X会发生什么？
3. 如果转置K会发生什么？

完整代码见 `cross_line_convolution.py` ,此处给出部分代码及相关结果
构建图像

```python
import torch

X = torch.ones([6, 6])
for i in range(len(X)):
  X[i, i] = 0
```
使用原卷积核的输出为：

```text
tensor([[-1.,  0.,  0.,  0.,  0.],
        [ 1., -1.,  0.,  0.,  0.],
        [ 0.,  1., -1.,  0.,  0.],
        [ 0.,  0.,  1., -1.,  0.],
        [ 0.,  0.,  0.,  1., -1.],
        [ 0.,  0.,  0.,  0.,  1.]])
```

注意到当前卷积核成功输出了水平边缘的检测结果，具体来说，是因为核 `[1, -1]` 计算的是水平方向上相邻元素的差值，即 `Y[i, j] = X[i, j] - X[i, j+1]`。这本质上是一个水平梯度算子，用于检测水平方向上的亮度变化（边缘）

- 当核覆盖从0到1的过渡时，输出为-1（例如，第0行第0列：0 - 1 = -1）。
- 当核覆盖从1到0的过渡时，输出为1（例如，第1行第0列：1 - 0 = 1）。
- 注意到输出比输入少了 `len(X) - len(K) + 1` 列是因为边缘检测时后续列无数据处理，而步幅(Stride)默认情况为1

转置 `X` 什么也不会发生，因为图像对称

转置 `K` 会使得卷积核的方向发生变化，从水平变为竖直，输出结果如下：

```text
tensor([[-1.,  1.,  0.,  0.,  0.,  0.],
        [ 0., -1.,  1.,  0.,  0.,  0.],
        [ 0.,  0., -1.,  1.,  0.,  0.],
        [ 0.,  0.,  0., -1.,  1.,  0.],
        [ 0.,  0.,  0.,  0., -1.,  1.]])
```

- 此时少行的原因同上

事实上，更通用、更完整的输出形状计算公式为：
$$H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[0] - \text{dilation}[0] \times (K_H - 1) - 1}{\text{stride}[0]} + 1\right\rfloor$$
$$W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[1] - \text{dilation}[1] \times (K_W - 1) - 1}{\text{stride}[1]} + 1\right\rfloor$$

其中，$H_{in}$ 和 $W_{in}$ 分别为输入的高度和宽度，$K_H$ 和 $K_W$ 分别为卷积核的高度和宽度，$\text{padding}$ 为填充，$\text{dilation}$ 为膨胀，$\text{stride}$ 为步幅。

####  如何通过改变输入张量和卷积核张量，将互相关运算表示为矩阵乘法？

注意到代码中关于卷积的运算代码为:

```python
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y
```

需要注意的是，当前实现虽然直观但并非矩阵乘法的相关实现，它是通过循环遍历输入张量的每个元素，然后与卷积核张量的对应元素进行相乘，最后求和得到输出张量的每个元素。

为了将卷积运算表示为矩阵乘法，我们需要将输入张量和卷积核张量转换为矩阵形式。`pytorch` 的底层是通过 `im2col` 函数实现的

更一般的数学表述为：

假设存在输入张量 $\mathbf{X}$ 和一个卷积核 $\mathbf{K}$，考虑多通道和批量处理的情况：

- 输入 $\mathbf{X}$ 的形状为 $(B, C, H, W)$，其中 $B$ 是批量大小，$C$ 是输入通道数，$H$ 和 $W$ 是高度和宽度。
- 卷积核 $\mathbf{K}$ 的形状为 $(N, C, k_h, k_w)$，其中 $N$ 是输出通道数，$k_h$ 和 $k_w$ 是卷积核的高度和宽度。
- 步幅为 $(s_h, s_w)$，填充为 $(p_h, p_w)$（可选）。

输出形状计算：

输出高度 $H_{\text{out}} = \left\lfloor \frac{H + 2p_h - k_h}{s_h} \right\rfloor + 1$。
输出宽度 $W_{\text{out}} = \left\lfloor \frac{W + 2p_w - k_w}{s_w} \right\rfloor + 1$。
输出 $\mathbf{Y}$ 的形状为 $(B, N, H_{\text{out}}, W_{\text{out}})$。

`im2col` 操作：

对于每个样本 in batch $B$，将输入 $\mathbf{X}$ 转换为矩阵 $\mathbf{X}_{\text{col}} \in \mathbb{R}^{(C \cdot k_h \cdot k_w) \times (H_{\text{out}} \cdot W_{\text{out}})}$。
$\mathbf{X}_{\text{col}}$ 的每一列对应一个输出位置 $(i, j)$，是由输入中所有通道的局部窗口展平拼接而成。具体地，对于输出位置 $(i, j)$，局部窗口的起始坐标为：
$$h_{\text{start}} = i \cdot s_h - p_h, \quad w_{\text{start}} = j \cdot s_w - p_w$$
然后提取每个通道上的子张量，展平后拼接成一个列向量。

卷积核展平：

将卷积核 $\mathbf{K}$ 重塑为矩阵 $\mathbf{K}_{\text{mat}} \in \mathbb{R}^{N \times (C \cdot k_h \cdot k_w)}$，其中每一行对应一个输出通道的核展平（行优先）。

矩阵乘法：

计算输出矩阵 $\mathbf{Y}_{\text{flat}} = \mathbf{K}_{\text{mat}} \cdot \mathbf{X}_{\text{col}}$，其中 $\mathbf{Y}_{\text{flat}} \in \mathbb{R}^{N \times (H_{\text{out}} \cdot W_{\text{out}})}$。
如果考虑批量，通常对于每个样本单独计算，或者使用批量矩阵乘法。最终输出 $\mathbf{Y}$ 是通过重塑 $\mathbf{Y}_{\text{flat}}$ 得到的。

最终输出：

将 $\mathbf{Y}_{\text{flat}}$ 重塑为 $(N, H_{\text{out}}, W_{\text{out}})$，然后加上批量维度，得到 $(B, N, H_{\text{out}}, W_{\text{out}})$.

假设 stride = 1, padding = 0, dilation = 1, 输入 $$X\in \mathbb{R}^{3\times3} = \begin{bmatrix}1&2&3\\4&5&6\\7&8&9\end{bmatrix}$$ ,卷积核 $$K \in \mathbb{R}^{2\times 2} = \begin{bamtrix}1&2\\3&4\end{bmatrix}$$ 则：

- 卷积核 `K` 展平为 `K_mat`：

$$K_{mat} = \begin{bmatrix}1&2&3&4\end{bmatrix}$$

- 输入 `X` 转换为 `X_col`：

$$X_{col} = \begin{bmatrix}1&2&4&5\\ 2&3&5&6\\ 4&5&7&8\\ 5&6&8&9\end{bmatrix}^T$$

- 则矩阵乘法下的卷积运算可表示为：$$Y_{hat} = K_{mat}\cdot X_{col} = \begin{bmatrix}1&2&3&4\end{bmatrix}\begin{bmatrix}1&2&4&5\\ 2&3&5&6\\ 4&5&7&8\\ 5&6&8&9\end{bmatrix}^T = \begin{bmatrix}37&47&67 &77\end{bmatrix}$$

- 输出 `Y` 重塑为 $\begin{bmatrix}37&47\\67 &77\end{bmatrix}$：

#### 手工设计一些卷积核。
1. 二阶导数的核的形式是什么？

二阶导数用于测量函数的“曲率”或“变化率的变化率”。在图像处理中，它常用于边缘检测（能同时响应明到暗和暗到明的过渡）和锐化。
最著名的二阶导数算子是拉普拉斯算子 (Laplacian)，它是二阶导数的和：$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$。
其常用的离散卷积核形式如下：
- 四邻域拉普拉斯核 (4-Neighborhood)
这个核只考虑上下左右四个方向的像素。
$$K_{\text{laplacian}_4} = \begin{bmatrix}
0 & 1 & 0 \\
1 & -4 & 1 \\
0 & 1 & 0
\end{bmatrix}$$

原理：中心点的二阶导数可以近似为 (上下左右邻居的和) - 4 * 中心点。

- 八邻域拉普拉斯核 (8-Neighborhood)
这个核考虑所有八个方向的相邻像素，对对角线方向也进行了近似。
$$K_{\text{laplacian}_8} = \begin{bmatrix}
1 & 1 & 1 \\
1 & -8 & 1 \\
1 & 1 & 1
\end{bmatrix}$$

原理：(所有邻居的和) - 8 * 中心点。

将原始图像与拉普拉斯核进行互相关操作，得到二阶导数图像（通常称为拉普拉斯响应）。这个响应图在边缘处会出现过零交叉(Zero-crossing)。

2. 积分的核的形式是什么？

在离散领域，“积分”操作通常被“局部平均”或“模糊”所替代。一个理想的积分核应该对窗口内所有像素赋予相同的正权重，并且所有权重之和为 1（以避免改变图像的整体亮度水平）。
最简单的积分核就是均值滤波器 (Mean Filter / Box Filter)。
3x3 均值滤波核:
$$K_{\text{mean}} = \frac{1}{9} \begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}$$

原理：计算一个 3x3 窗口内所有像素的平均值，并用这个平均值替换中心像素。这相当于对该区域进行了一个非常粗略的“积分”近似。核越大（如5x5, 7x7），模糊（积分）效果越强。

高斯核 (Gaussian Filter) 是一种更优秀的“积分”核，它更符合自然现象，因为它给中心点赋予最高权重，权重随着距离中心点的增加而平滑下降。
$$G_{\sigma} = \frac{1}{159} \begin{bmatrix}
2 & 4 & 5 & 4 & 2 \\
4 & 9 & 12 & 9 & 4 \\
5 & 12 & 15 & 12 & 5 \\
4 & 9 & 12 & 9 & 4 \\
2 & 4 & 5 & 4 & 2 \\
\end{bmatrix}$$
这是一个近似化的 5x5 高斯核 ($\sigma \approx 1.0$)，其所有权重之和为 1。

高斯核对边缘检测效果更好。

3. 得到d次导数的最小核的大小是多少？

计算 d 阶导数所需的最小卷积核大小是 d + 1。
类似于数学归纳：

- 一阶导数：最小需要 2 个点。
  例如前向差分：$f'(x) \approx \frac{f(x+h) - f(x)}{h}$。这可以用一个 1x2 的核 [-1, 1] 来实现。
- 二阶导数：最小需要 3 个点。
  中心差分：$f''(x) \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2}$。这对应一个 1x3 的核 [1, -2, 1]。这正是拉普拉斯核在一维上的形式。
- 推广到 d 阶导数：为了估算第 d 阶导数，你需要足够多的点来构建一个能够精确拟合局部多项式（通常是 d 次多项式）的差分方程。这至少需要 d+1 个不同的点（采样值）。

因此，一个一维的、离散的、用于计算 d 阶导数的卷积核，其最小长度为 d + 1。
多维则考虑是否要求各向同性，在各向同性的情况下，最小尺寸为 $(d+1)^2$

## 6.3 填充和步幅
### Exercises 6.3
#### 对于音频信号，步幅2说明什么？

Stride（步幅）大于 1 的操作在效果上是一种“稀疏采样”或“降采样”,同时由于它跳过了某些位置，它对输入特征的微小位置变化变得不那么敏感，扩大了感受野，使得模型更关注于某个区域内是否存在某种特征，而不是它精确出现在哪个样本点上。

#### 步幅大于1的计算优势是什么？

见上一问题，意义在普适问题上其实差不多，细化方面会有些许区别

## 6.4 多通道和批量处理

### Exercises 6.4

#### 假设我们有两个卷积核，大小分别为 $k_1$ 和 $k_2$ （中间没有非线性激活函数）。
1. 证明运算可以用单次卷积来表示。
  在CNN中，卷积是一种线性操作。连续应用两个卷积核（无非线性激活函数）相当于一个单一的卷积操作。数学上，设第一个卷积核为 $W_1$（形状为 $[C_{\text{mid}}, C_{\text{in}}, k_1, k_1]$），第二个卷积核为 $W_2$（形状为 $[C_{\text{out}}, C_{\text{mid}}, k_2, k_2]$）。对输入 $X$ 先应用 $W_1$ 得到中间输出 $Y = \text{conv}(X, W_1)$，再应用 $W_2$ 得到最终输出 $Z = \text{conv}(Y, W_2)$。
  由于卷积的线性性质，有：
  $$Z = \text{conv}(\text{conv}(X, W_1), W_2) = \text{conv}(X, W_{\text{combined}})$$
  其中 $W_{\text{combined}}$ 是等效的单个卷积核，通过对 $W_1$ 和 $W_2$ 在空间维度上进行卷积得到。具体地，对于每个输出通道 $c_{\text{out}}$ 和输入通道 $c_{\text{in}}$，等效核的元素是 $W_2$ 与 $W_1$ 的卷积结果：
  $$W_{\text{combined}}[c_{\text{out}}, c_{\text{in}}, :, :] = \sum_{m=1}^{C_{\text{mid}}} \text{conv}(W_2[c_{\text{out}}, m, :, :], W_1[m, c_{\text{in}}, :, :])$$
  这里，$\text{conv}$ 表示二维卷积操作（无填充，步长1）。因此，整个运算可以被单次卷积表示。

2. 这个等效的单个卷积核的维数是多少呢？
  等效卷积核 $W_{\text{combined}}$ 的维数（形状）取决于原始核的大小和通道数。空间尺寸上，如果两个核的大小分别为 $k_1 \times k_1$ 和 $k_2 \times k_2$，则等效核的空间大小为 $k_{\text{combined}} \times k_{\text{combined}}$，其中 $k_{\text{combined}} = k_1 + k_2 - 1$（这是卷积操作中输出尺寸的计算公式，假设无填充和步长1）。
  在通道维度上：

  输入通道数：$C_{\text{in}}$
  输出通道数：$C_{\text{out}}$
  因此，等效核的形状为 $[C_{\text{out}}, C_{\text{in}}, k_{\text{combined}}, k_{\text{combined}}]$

  例如，如果 $k_1 = 3$, $k_2 = 3$, $C_{\text{in}} = 1$, $C_{\text{mid}} = 2$, $C_{\text{out}} = 1$，则等效核的大小为 $5 \times 5$（因为 $3 + 3 - 1 = 5$），形状为 $[1, 1, 5, 5]$。

3. 反之亦然吗？
  反之不亦然。并非所有卷积核都能被精确分解为两个较小核的卷积。这是因为卷积核的分解相当于求解一个线性系统，其可行性取决于核的秩和结构。例如，如果一个卷积核的大小为 $k \times k$，它可能无法被分解为两个核大小为 $k_1 \times k_1$ 和 $k_2 \times k_2$ 的乘积（其中 $k_1 + k_2 - 1 = k$），除非核满足特定条件（如可分离性）。

#### 假设输入形状为 $c_i \times h \times w$，卷积核是形状为 $c_o \times c_i \times k_h \times k_w$，填充为 $(p_h, p_w)$，步幅为 $(s_h, s_w)$。

**1. 前向传播的计算成本（乘法和加法）是多少？**
  前向传播涉及卷积操作，每个输出元素的计算需要与卷积核窗口进行点积。计算成本以乘法和加法次数衡量。
  乘法次数：每个输出元素需要 $c_i \times k_h \times k_w$ 次乘法（因为每个输入通道和每个核元素相乘）。总乘法次数为：
  $$\text{Multiplications} = c_o \times h_o \times w_o \times c_i \times k_h \times k_w$$
  加法次数：每个输出元素需要 $c_i \times k_h \times k_w - 1$ 次加法（用于累加乘积）。总加法次数为：
  $$\text{Additions} = c_o \times h_o \times w_o \times (c_i \times k_h \times k_w - 1)$$
  注意：在实际实现中，加法可能被优化，但这是理论值。总乘加操作数（MAC）为 $c_o \times h_o \times w_o \times c_i \times k_h \times k_w$，常用于衡量计算复杂度。
**2. 内存占用是多少？**
  - 输入激活值：$c_i \times h \times w$ 元素
  - 输出激活值：$c_o \times h_o \times w_o$ 元素
  - 权重参数：$c_o \times c_i \times k_h \times k_w$ 元素
  总内存占用为：
  $$\text{Memory}_{\text{forward}} = c_i \times h \times w + c_o \times h_o \times w_o + c_o \times c_i \times k_h \times k_w$$
  在训练时，这些值需要存储用于反向传播。  
**3. 反向传播的内存占用是多少？**
  反向传播需要存储前向传播的激活值、梯度值以及参数。峰值内存占用包括：
  存储的前向激活值：输入激活值（$c_i \times h \times w$）和输出激活值（$c_o \times h_o \times w_o$）
  梯度值：输出梯度（来自上一层，大小 $c_o \times h_o \times w_o$）、输入梯度（计算后传递下一层，大小 $c_i \times h \times w$）、权重梯度（计算后用于更新，大小 $c_o \times c_i \times k_h \times k_w$）
  权重参数：$c_o \times c_i \times k_h \times k_w$（已存储）
  总内存占用约为：
  $$\text{Memory}_{\text{backward}} = 2 \times (c_i \times h \times w) + 2 \times (c_o \times h_o \times w_o) + 2 \times (c_o \times c_i \times k_h \times k_w)$$
  这表示大约两倍的激活值和两倍的参数内存。实际占用可能因实现优化而略有不同。
**4. 反向传播的计算成本是多少？**
  反向传播包括计算权重梯度和输入梯度，计算成本类似前向传播但操作数更高。
  权重梯度计算：
  乘法次数：每个权重元素需要 $h_o \times w_o$ 次乘法（与输出梯度相乘）。总乘法次数为：
  $$\text{Multiplications}_{\text{weight grad}} = c_o \times c_i \times k_h \times k_w \times h_o \times w_o$$
  加法次数：每个权重元素需要 $h_o \times w_o - 1$ 次加法（求和）。总加法次数为：
  $$\text{Additions}_{\text{weight grad}} = c_o \times c_i \times k_h \times k_w \times (h_o \times w_o - 1)$$
  输入梯度计算：
  乘法次数：每个输入元素需要 $c_o \times k_h \times k_w$ 次乘法（与权重相乘）。总乘法次数为：
  $$\text{Multiplications}_{\text{input grad}} = c_i \times h \times w \times c_o \times k_h \times k_w$$
  加法次数：每个输入元素需要 $c_o \times k_h \times k_w - 1$ 次加法（求和）。总加法次数为：
  $$\text{Additions}_{\text{input grad}} = c_i \times h \times w \times (c_o \times k_h \times k_w - 1)$$
  总反向传播计算成本：
  总乘法次数：
  $$\text{Total Multiplications} = c_o \times c_i \times k_h \times k_w \times h_o \times w_o + c_i \times h \times w \times c_o \times k_h \times k_w$$
  注意：由于 $c_o \times h_o \times w_o \times c_i \times k_h \times k_w = c_i \times h \times w \times c_o \times k_h \times k_w$（因为输出尺寸与输入尺寸相关），但通常简化表示为 $2 \times c_o \times h_o \times w_o \times c_i \times k_h \times k_w$，但精确值如上。
  总加法次数：
  $$\text{Total Additions} = c_o \times c_i \times k_h \times k_w \times (h_o \times w_o - 1) + c_i \times h \times w \times (c_o \times k_h \times k_w - 1)$$

#### 如果我们将输入通道c i 和输出通道c o 的数量加倍，计算数量会增加多少？如果我们把填充数量翻一番会怎么样？

计算成本主要以乘加操作数（MAC）衡量，基于前向传播的公式：
$$\text{Computations} = c_o \times h_o \times w_o \times c_i \times k_h \times k_w$$
其中，输出尺寸 $h_o$ 和 $w_o$ 取决于输入尺寸、卷积核尺寸、填充和步幅：
$$h_o = \left\lfloor \frac{h + 2p_h - k_h}{s_h} \right\rfloor + 1, \quad w_o = \left\lfloor \frac{w + 2p_w - k_w}{s_w} \right\rfloor + 1$$

如果将输入通道 $c_i$ 和输出通道 $c_o$ 的数量加倍，计算数量会增加多少？

- 原始计算成本：
$$\text{Computations}_{\text{original}} = c_o \times h_o \times w_o \times c_i \times k_h \times k_w$$

新参数：将 $c_i$ 和 $c_o$ 加倍后，新输入通道为 $2c_i$，新输出通道为 $2c_o$。输出尺寸 $h_o$ 和 $w_o$ 不变，因为它们与通道数无关。
新计算成本：
$$\text{Computations}_{\text{new}} = (2c_o) \times h_o \times w_o \times (2c_i) \times k_h \times k_w = 4 \times c_o \times h_o \times w_o \times c_i \times k_h \times k_w$$

增加比例：计算数量增加为原来的4倍（即增加300%）。这是因为通道数加倍在计算成本中具有乘积效应。

- 如果把填充数量翻一番会怎么样？
填充数量翻一番意味着新填充为 $(2p_h, 2p_w)$。这会影响输出尺寸 $h_o$ 和 $w_o$，从而影响计算成本。

原始输出尺寸：
$$h_o = \left\lfloor \frac{h + 2p_h - k_h}{s_h} \right\rfloor + 1, \quad w_o = \left\lfloor \frac{w + 2p_w - k_w}{s_w} \right\rfloor + 1$$

新输出尺寸：
$$h_o' = \left\lfloor \frac{h + 4p_h - k_h}{s_h} \right\rfloor + 1, \quad w_o' = \left\lfloor \frac{w + 4p_w - k_w}{s_w} \right\rfloor + 1$$

新计算成本：
$$\text{Computations}_{\text{new}} = c_o \times h_o' \times w_o' \times c_i \times k_h \times k_w$$

增加比例：计算数量的增加取决于 $h_o'$ 和 $w_o'$ 相对于 $h_o$ 和 $w_o$ 的变化。具体比例无法一概而论，因为它受输入尺寸 $h, w$、卷积核尺寸 $k_h, k_w$、步幅 $s_h, s_w$ 的影响。但通常，填充增加会使输出尺寸增大，从而增加计算成本。

示例假设：如果步幅 $s_h = s_w = 1$，且输入尺寸较大，则 $h_o' \approx h_o + 2p_h$ 和 $w_o' \approx w_o + 2p_w$（近似）。因此，计算成本增加比例约为 $\frac{(h_o + 2p_h)(w_o + 2p_w)}{h_o w_o}$。
一般情况：填充翻倍会使输出尺寸增加，但增加幅度非线性。实际中，填充翻倍通常会导致计算成本增加，但具体倍数需要根据参数计算。

#### 如果卷积核的高度和宽度是 $k_h = k_w = 1$，前向传播的计算复杂度是多少？

一般卷积层的前向传播计算复杂度（以乘加操作数，MAC衡量）为：
$$\text{Computations} = c_o \times h_o \times w_o \times c_i \times k_h \times k_w$$
其中：

$c_o$ 是输出通道数
$h_o$ 和 $w_o$ 是输出特征图的高度和宽度
$c_i$ 是输入通道数
$k_h$ 和 $k_w$ 是卷积核的高度和宽度

代入 $k_h = k_w = 1$：
$$\text{Computations} = c_o \times h_o \times w_o \times c_i \times 1 \times 1 = c_o \times h_o \times w_o \times c_i$$
因此，计算复杂度简化为了 $c_o \times h_o \times w_o \times c_i$。

####  当卷积窗口不是1 × 1时，如何使用矩阵乘法实现卷积？

[笔记](https://vermosh.top/brainstorm-time/d2l-pytorch/chapter-6-cnn-1/)中有详细推导,此处不再赘述