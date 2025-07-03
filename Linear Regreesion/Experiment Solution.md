# Linear Neural Networks for Classification / 线性神经网络

## Object-Oriented Design for Implementation / 从零开始の线性回归

1. **如果我们将权重初始化为零，会发生什么。算法仍然有效吗？**
  (***fix***:针对单层线性回归)显然是有效的。这是因为在权重为0的情况下，模型初始化为一条垂直于y轴的的直线(或者说，平行于某一轴超平面的超平面)，此时梯度仍然可以正常更新。
    > 若为多层线性回归又会发生什么？
2. **假设试图为电压和电流的关系建立一个模型。自动微分可以用来学习模型的参数吗?**
  针对于线性元件，有欧姆定律:\[U=IR\]显然是可以的；
  非线性元件则不一定适用(线性可微与否)
3. **能基于普朗克定律使用光谱能量密度来确定物体的温度吗？**
  普朗克定律的能量密度频谱形式：\[\frac{8\pi h \nu^3}{c^3}=e^{\frac{h\nu}{kT}-1}\]注意到 \(\nu^3\propto e^{\frac{h\nu}{kT}}\) 因而可以。
4. **计算二阶导数时可能会遇到什么问题？这些问题可以如何解决？**
    - **可能的问题**
      - **计算开销较大**：二阶导(*Hessian Matrix*)的空间复杂度为\(O(n^2)\)
      - **数值不稳定**：二阶导对噪声敏感，易出现 ***NAN***
      - **代码嵌套**：需要 `pytorch` 的 `autograd` 多次反向传播
    - **解决方法**
      - 采用**牛顿法**近似
5. **为什么在`squared_loss`函数中需要使用`reshape`函数？**
   防止 \(\hat{y}\) 与 \(y\) 维度不一意外引发**广播机制**
6. **尝试使用不同的学习率，观察损失函数值下降的快慢。**
   - LearningRate = 10 (这显然过大，此处只为演示例)

    ```text
      epoch 1, loss nan
      epoch 2, loss nan
      epoch 3, loss nan
      learned w: tensor([nan, nan], requires_grad=True), true w: tensor([1.0000, 1.2000]),error w: tensor([nan, nan], grad_fn=<SubBackward0>)
      learned b: tensor([nan], requires_grad=True), true b: 2.5,error b: tensor([nan], grad_fn=<SubBackward0>)
    ```

    观察到损失函数 ***NAN***
   - LearningRate = 0.0001
  
    ```text
      epoch 1, loss 4.121013
      epoch 2, loss 4.041764
      epoch 3, loss 3.964042
      learned w: tensor([0.0171, 0.0452], requires_grad=True), true w: tensor([1.0000, 1.2000]),error w: tensor([0.9829, 1.1548], grad_fn=<SubBackward0>)
      learned b: tensor([0.0731], requires_grad=True), true b: 2.5,error b: tensor([2.4269], grad_fn=<SubBackward0>)
    ```

    观察到收敛过慢(截至结束前，残差仍 > 1)
7. **如果样本个数不能被批量大小整除，data_iter函数的行为会有什么变化？**
  `min(i + batch_size, num_examples)` 会截断最后不足一个批量的样本
  