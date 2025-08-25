def mmd_test_torch(X_train, X_test, device='cpu', alpha=0.05):
    """
    使用PyTorch计算MMD并进行假设检验。
    
    参数:
    X_train (np.ndarray): 训练集样本
    X_test (np.ndarray): 测试集样本
    device (str): 'cpu' 或 'cuda'
    alpha (float): 显著性水平
    
    返回:
    mmd_value (float): MMD统计量
    p_value (float): 检验的p值（通过置换检验计算）
    rejected (bool): 是否拒绝原假设（是否存在显著偏移）
    """
    
    # 转换为PyTorch张量
    X = torch.FloatTensor(X_train).to(device)
    Y = torch.FloatTensor(X_test).to(device)
    
    # 高斯核函数
    def gaussian_kernel(X, Y, sigma=1.0):
        """
        计算高斯核矩阵
        """
        X_norm = torch.sum(X**2, dim=1, keepdim=True)
        Y_norm = torch.sum(Y**2, dim=1, keepdim=True)
        
        # 计算 pairwise 距离的平方
        dist_sq = X_norm + Y_norm.t() - 2 * torch.mm(X, Y.t())
        
        # 应用高斯核
        return torch.exp(-dist_sq / (2 * sigma**2))
    
    # 计算MMD²的无偏估计
    def mmd2_unbiased(X, Y, sigma=1.0):
        m = X.size(0)
        n = Y.size(0)
        
        # 计算核矩阵
        K_XX = gaussian_kernel(X, X, sigma)
        K_YY = gaussian_kernel(Y, Y, sigma)
        K_XY = gaussian_kernel(X, Y, sigma)
        
        # 对角线元素置零（用于无偏估计）
        K_XX.fill_diagonal_(0)
        K_YY.fill_diagonal_(0)
        
        # 计算MMD²无偏估计
        term1 = torch.sum(K_XX) / (m * (m - 1))
        term2 = torch.sum(K_YY) / (n * (n - 1))
        term3 = 2 * torch.sum(K_XY) / (m * n)
        
        return term1 + term2 - term3
    
    # 计算原始MMD值
    mmd_value = mmd2_unbiased(X, Y).item()
    
    # 使用置换检验计算p值
    n_permutations = 1000
    count = 0
    
    # 合并所有样本
    Z = torch.cat([X, Y], dim=0)
    
    for i in range(n_permutations):
        # 随机打乱顺序
        idx = torch.randperm(Z.size(0))
        Z_perm = Z[idx]
        
        # 分割为两组
        X_perm = Z_perm[:X.size(0)]
        Y_perm = Z_perm[X.size(0):]
        
        # 计算置换后的MMD值
        mmd_perm = mmd2_unbiased(X_perm, Y_perm).item()
        
        # 统计大于原始MMD值的次数
        if mmd_perm >= mmd_value:
            count += 1
    
    # 计算p值
    p_value = (count + 1) / (n_permutations + 1)  # 加1是为了避免p值为0
    
    # 做出决策
    rejected = p_value < alpha
    
    print(f"MMD值: {mmd_value:.6f}")
    print(f"p值: {p_value:.4f}")
    if rejected:
        print(f"在显著性水平 {alpha} 下拒绝原假设：存在显著的协变量偏移！")
    else:
        print(f"在显著性水平 {alpha} 下无法拒绝原假设：未检测到显著偏移。")
    
    return mmd_value, p_value, rejected

# 使用示例
# mmd, p_val, rejected = mmd_test_torch(train_features, test_features)
