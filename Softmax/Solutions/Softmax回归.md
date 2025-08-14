# Softmax回归与相关问题分析 (Softmax Regression and Related Problems)

## 1. 指数族与Softmax的联系 (Exponential Family and Softmax Connection)

### 1.1 计算softmax交叉熵损失的二阶导数

给定损失函数：
$$l(\mathbf{y},\hat{\mathbf{y}}) = -\sum_j y_j \ln \hat{y}_j$$

一阶导数为：
$$\partial_{o_j} l(y, \hat{y}) = \text{softmax}(o_j) - y_j$$

二阶导数为：
$$\partial^2_{o_i o_j} l(y, \hat{y}) =
\begin{cases}
p_i(1-p_i) & i = j \\
-p_i p_j & i \neq j
\end{cases}$$
其中 $p_j = \text{softmax}(o_j)$

Hessian矩阵形式：
$$H = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$$

### 1.2 分布方差与二阶导数的关系

softmax输出的方差：
$$\text{Var}(\mathbf{p}) = \mathbb{E}[p_j^2] - (\mathbb{E}[p_j])^2 = \sum_{j=1}^q p_j^2 - \frac{1}{q}$$

**性质分析**：
- 当 $\mathbf{o}$ 尺度增大：输出趋近one-hot分布，方差 $\to 1 - \frac{1}{q}$
- 当 $\mathbf{o}$ 尺度减小：输出趋近均匀分布，方差 $\to 0$
- 方差与熵负相关：方差越大，分布越尖锐（熵越小）

---

## 2. 类别编码问题 (Class Encoding Problem)

### 2.1 二进制编码的问题

当三类概率相等 $(1/3, 1/3, 1/3)$ 时：
- **编码效率低**：需要2比特表示3种状态，理论效率应为 $\log_2 3 \approx 1.585$ 比特/符号，实际效率仅1.5
- **冗余问题**：4种编码组合中1种未使用（如11），造成资源浪费
- **解码歧义**：未使用编码可能导致错误解析

### 2.2 改进编码方案

**联合编码 $n$ 个观测**：
- 可能结果数：$3^n$ 种等概率组合
- 所需比特：$\lceil n \log_2 3 \rceil$
- **渐进最优性**：当 $n \to \infty$ 时，平均比特数 $\to \log_2 3 \approx 1.585$，达到信息论极限

---

## 3. RealSoftMax 性质分析 (RealSoftMax Properties)

### 3.1 证明 $\mathrm{RealSoftMax}(a, b) > \max(a, b)$

设 $m = \max(a,b)$：
$$
\begin{aligned}
\mathrm{RealSoftMax}(a,b) &= \log(e^a + e^b) \\
&= m + \log(e^{a-m} + e^{b-m}) > m
\end{aligned}
$$
因 $\log(e^{a-m} + e^{b-m}) > 0$（至少一项为1，另一项≥0）

### 3.2 证明 $\lambda^{-1}\mathrm{RealSoftMax}(\lambda a,\lambda b) > \max(a,b)$ ($\lambda>0$)

$$
\begin{aligned}
\lambda^{-1}\mathrm{RealSoftMax}(\lambda a,\lambda b) &= \lambda^{-1}\log(e^{\lambda a} + e^{\lambda b}) \\
&= \max(a,b) + \lambda^{-1}\log(e^{\lambda(a-m)} + e^{\lambda(b-m)}) > \max(a,b)
\end{aligned}
$$
因指数和 $>1$，对数项 $>0$

### 3.3 证明 $\lim_{\lambda\to\infty} \lambda^{-1}\mathrm{RealSoftMax}(\lambda a,\lambda b) = \max(a,b)$

假设 $a > b$：
$$
\begin{aligned}
\lim_{\lambda\to\infty} \lambda^{-1}\log(e^{\lambda a} + e^{\lambda b})
&= \lim_{\lambda\to\infty} \left[a + \lambda^{-1}\log(1 + e^{-\lambda(a-b)})\right] \\
&= a + 0 = \max(a,b)
\end{aligned}
$$

### 3.4 soft-min 形式

$$\mathrm{softmin}(a,b) = -\mathrm{RealSoftMax}(-a,-b) = -\log(e^{-a} + e^{-b})$$

**性质**：
- $\mathrm{softmin}(a,b) < \min(a,b)$
- $\lim_{\lambda\to\infty} \lambda^{-1}\mathrm{softmin}(\lambda a,\lambda b) = \min(a,b)$

### 3.5 多变量扩展

对于 $n$ 个变量 $x_1,...,x_n$：
$$\mathrm{RealSoftMax}(x_1,...,x_n) = \log\left(\sum_{i=1}^n e^{x_i}\right)$$

**性质一致**：
1. 结果 $> \max(x_i)$
2. $\lambda^{-1}\mathrm{RealSoftMax}(\lambda x_1,...,\lambda x_n) > \max(x_i)$ ($\lambda>0$)
3. $\lim_{\lambda\to\infty} \lambda^{-1}\mathrm{RealSoftMax}(\lambda x_1,...,\lambda x_n) = \max(x_i)$
4. $\mathrm{softmin}(x_1,...,x_n) = -\mathrm{RealSoftMax}(-x_1,...,-x_n)$
