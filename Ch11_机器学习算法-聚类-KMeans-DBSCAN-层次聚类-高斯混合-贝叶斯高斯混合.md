

# 聚类

## KMeans

- 算法原理:
    - 指定k个质心, 计算全部样本与质心的欧氏距离, 分别将与质心距离近的样本划分为簇(归类)
    - 计算各个簇的中心, 将中心更新为新质心
    - 再计算所有样本与新质心的距离, 再划分簇(归类)
    - 再更新质心位置到簇中心, 直到质心位置不怎么变化就停止
- WCSS簇内误差平方和公式(其中k为质心数量, C_i为第几个簇, x_i为簇内第几个样本, mu_i为第几个簇的质心)(计算簇内各个样本与质心的距离的平方和, 再将各个簇求和)
    
    $$
    WCSS=\sum^k_{i=1} \sum _{x\in C_i} (x -\mu_i)^2
    $$
    
- 计算划分完簇(归类后), 簇内的样本与质心的距离的平方和, 如果质心离簇中心较远, 那么距离较长, 簇内平方和的值也会较大, 多次迭代质心位置, 将质心更新为最佳簇内中心位置, 簇内平方和的值会下降
- 导入KMeans语句
    
    ```python
    from sklearn.cluster import KMeans
    ```
    
- KMeans参数详解
    
    ```python
    KMeans(
        n_clusters=8, #质心数k
        init='k-means++', #质心初始化方法, 默认k-means++智能初始化, 备选参数random随机选择样本点(超大数据集), 可自定义质心坐标数组[n_clusters, n_features]
        n_init='auto', #当init='k-means++'时, n_init=10, 当init='random'时, n_init=1, 可自定义多次运行取最佳结果(选择WCSS最小的那次)
        max_iter=300, #单次运行的最大迭代次数(计算距离+归类+更新质心位置的迭代次数)(简单数据集100-300, 复杂数据集500-1000)
        tol=0.0001, #每次迭代使WCSS下降的阈值, 值越小精度越高收敛越慢
        verbose=0, #输出过程, 0无输出, 1输出进度和WCSS, 2详细输出每次迭代的计算数据
        random_state=42,
        copy_x=True,
        algorithm='lloyd' #算法优化, 备选参数elkan加速收敛, (推荐小数据用elkan, 大数据用lloyd)
    )
    ```
    
- 拟合数据
    
    ```python
    KM = KMeans().fit(X)
    ```
    
- 获取每个簇的标签
    
    ```python
    KM.labels_
    ```
    
- 获取最终质心坐标数组
    
    ```python
    KM.cluster_centers_
    ```
    
- 获取最终WCSS值
    
    ```python
    KM.inertia_
    ```
    
- 获取实际运行的迭代次数
    
    ```python
    KM.n_iter_
    ```
    
- 获取KMeans参数的函数
    
    ```python
    from sklearn.cluster import k_means
    k_means(X, 质心数, return_n_iter=True)
    #函数返回质心坐标数组, 每个簇的标签, WCSS值, 实际运行的迭代次数
    ```
    
- 质心数k—超参数确定(通过计算每个簇轮廓系数)
    - 指定k值区间, for循环遍历所有k值, 计算轮廓系数, 找出最高轮廓系数所对应的k值, 最好是结合WCSS曲线综合判断
        
        ```python
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        def find_best_k(X, Kmax):
            k_scores = []
            
            #K从2到Kmax
            for k in range(2, Kmax):
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = kmeans.fit_predict(X)
                
                # 计算轮廓系数(确保至少有两个簇)
                score = silhouette_score(X, labels)
                
                k_scores.append(score)
                print(f"K={k} \t 轮廓系数={score:.4f}")
            
            # 找到最佳K值
            best_idx = np.argmax(k_scores)
            best_k = best_idx + 2  #因为K从2开始
            best_score = k_scores[best_idx]
            
            print("\n" + "="*50)
            print(f"最佳K值: {best_k} (轮廓系数={best_score:.4f})")
            print("="*50)
            
            return best_k, best_score, k_scores
        ```
        
- 最佳质心数k诊断
    
    ```python
    import numpy as np
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn.cluster import KMeans
    
    def cluster_diagnosis(X, best_k):
        # 1. 使用最佳K值运行KMeans
        kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # 2. 计算轮廓系数
        avg_silhouette = silhouette_score(X, cluster_labels) #理想轮廓系数>0.5
        sample_silhouette = silhouette_samples(X, cluster_labels) #单个样本轮廓系数
        
        # 3. 分析簇大小
        cluster_sizes = [np.sum(cluster_labels == i) for i in range(best_k)] #计算各个簇的样本数量
        min_size = min(cluster_sizes) #取簇内样本数量最大值
        max_size = max(cluster_sizes) #取簇内样本数量最小值
        size_ratio = max_size / min_size if min_size > 0 else float('inf') #最大样本数的簇与最小样本数簇的比例, 应尽量均衡为1:1
        
        # 4. 检查负轮廓系数样本
        negative_samples = np.sum(sample_silhouette < 0) #负轮廓系数(聚类错误的样本数)
        negative_ratio = negative_samples / len(X) #判断错误的样本数占总样本数的比例应小于5%
        
        # 5. 计算簇内离散度
        intra_cluster_dispersion = kmeans.inertia_ / len(X) #离散度越小越好
        
        # 6. 诊断报告
        report = {
            "best_k": best_k,
            "avg_silhouette": avg_silhouette,
            "min_cluster_size": min_size,
            "max_cluster_size": max_size,
            "size_ratio": size_ratio,
            "negative_samples": negative_samples,
            "negative_ratio": negative_ratio,
            "intra_cluster_dispersion": intra_cluster_dispersion,
            "cluster_sizes": cluster_sizes,
            "silhouette_scores": sample_silhouette
        }
        
        return report
    ```
    

## DBSCAN

- 算法原理
    - 计算所有数据点, 在指定的半径范围(超参数eps)内包含自身点的最少邻居数(超参数min_samples), 如果半径范围内的数据点数量超过最少邻居数, 那么这个区域为较高数据密度, 如果半径范围内的数据点少于最少邻居数, 那么这个区域的点为数据边界点(排除噪声点), 反复计算最终形成同类数据簇轮廓(可能为不规则的轮廓)
    - 计算结果为正整数从0开始的类别数, 以及-1值(表示噪声点, 可用作检测异常值)
- 导入DBSCAN语句
    
    ```python
    from sklearn.cluster import DBSCAN
    ```
    
- DBSCAN参数详解
    
    ```python
    DBSCAN(
        eps=0.5, #半径超参数
        min_samples=5, #最少邻居数超参数
        metric='euclidean', #距离计算方式
        metric_params=None, #距离计算额外参数
        algorithm='auto', #算法
        leaf_size=30, #树结构的叶子大小
        p=None, #当metric='minkowski'时有效, p=1为曼哈顿距离, p=2为欧式距离
        n_jobs=-1
    )
    ```
    
- 拟合训练数据
    
    ```python
    DBS = DBSCAN().fit(X)
    ```
    
- 获取每个簇的标签
    
    ```python
    DBS.labels_ #返回值中-1为噪声点(不属于任何一个簇), 其他和KMeans一样
    ```
    
- 获取所有核心点在特征矩阵中的行索引
    
    ```python
    DBS.core_sample_indices_
    ```
    
- 获取所有核心点的特征值数组
    
    ```python
    DBS.components_ #返回作为核心点的那些样本的特征值
    ```
    
- DBSCAN超参数调优方法
    - 先确定min_samples=2x数据维度, k=min_samples-1(排除自身), 计算每个数据点到k个最近邻的距离, 将计算出的距离最后一列(使半径范围内包含min_samples个点的最远距离点)升序排序作为纵轴, 样本数据序列作为横轴, 绘制可视化图, 找到曲线斜率变化幅度较大的位置为拐点, 取距离分位数在95%位置的数为半径eps(如果聚类噪声点过多或簇过大过小可在95%分位数附近微调)
    - 使用NearestNeighbors对超参数调优的语句
    
    ```python
    from sklearn.neighbors import NearestNeighbors
    min_samples = 2*'数据维度'
    k = min_samples - 1  #不包括自身
    NN = NearestNeighbors(n_neighbors=k).fit(X)
    distances, index = NN.kneighbors(X) #distances距离数组的列数等于k, 从近到远从左往右排序, 第一列为0是和自己的距离
    sorted_k_distances = np.sort(distances[:, -1]) #取包含min_samples个点的最远距离点的最后一列, 并升序排列
    eps = np.percentile(distances[:, -1],0.95) 
    ```
    

## AgglomerativeClustering

- 层次聚类: 将每个样本视为一个独立的簇, 重复合并距离最近的两个簇, 直到所有样本聚为一个簇或达到指定簇数量(适用于文档主题层次结构挖掘)
- 导入AgglomerativeClustering语句
    
    ```python
    from sklearn.cluster import AgglomerativeClustering
    ```
    
- AgglomerativeClustering参数详解
    
    ```python
    AgglomerativeClustering(
        n_clusters=2, #指定最终聚类数量
        metric='euclidean', #距离度量方法, 备选参数manhattan曼哈顿距离, cosine余弦相似度
        linkage='ward', #备选参数average集合cosine处理高维数据
        memory=None,
        connectivity=None, #约束聚类过程(只合并相邻样本)
        compute_full_tree='auto', #是否计算完整的层次树
        compute_distances=False, #是否计算簇合并时的距离
        distance_threshold=None #指定聚类停止的距离阈值(簇间距离大于阈值时停止合并)
    )
    ```
    
- 拟合训练数据
    
    ```python
    AgglomerativeClustering().fit(X)
    ```
    
- 查看聚合后的标签
    
    ```python
    AC.labels_
    ```
    
- 查看聚合后的簇数量
    
    ```python
    AC.n_clusters_
    ```
    

## GaussianMixture

- 高斯混合模型：指定超参数n_components(要聚合的簇数量), 算法会使用EM算法计算每个样本(数据点)属于每个簇的概率, 并以最大概率来归类; 同时指定n_init参数来使算法重复运行计算以防止EM算法收敛到局部最优解
- 导入GaussianMixture语句
    
    ```python
    from sklearn.mixture import GaussianMixture
    ```
    
- GaussianMixture参数详解
    
    ```python
    GaussianMixture(
         n_components=1, #聚合簇参数
         covariance_type='full', #簇形状约束(full:每个簇任意方向和形状; tied:每个簇大小形状相同; diag:簇的坐标轴与全局坐标轴平行; spherical: 簇形状为圆形, 但大小不同)
         tol=0.001, #收敛阈值
         reg_covar=1e-6, #协方差对角线的正则化项, 防止协方差矩阵奇异（不可逆），确保数值稳定性
         max_iter=100, #最大迭代次数(即使未达到收敛阈值，最多迭代这么多次)
         n_init=1, #算法重复初始化计算次数(防止陷入局部最优，选择最佳初始化)
         init_params='kmeans', #初始化方法
         weights_init=None, #自定义初始化权重
         means_init=None, #自定义初始化均值
         precisions_init=None, #自定义初始化协方差矩阵的逆
         random_state=42,
         warm_start=False, #热启动
         verbose=0, #日志输出
         verbose_interval=10 #日志输出间隔迭代数
    )
    ```
    
- 拟合训练数据
    
    ```python
    GM.fit(x)
    ```
    
- 查看训练过程的收敛情况
    
    ```python
    GM.converged_ #返回布尔值，如果达到收敛阈值为True，否则为False
    ```
    
- 查看模型收敛的实际迭代次数
    
    ```python
    GM.n_iter_
    ```
    
- 评估模型拟合优度
    
    ```python
    GM.lower_bound_ #值越大通常意味着模型对数据的拟合越好
    ```
    
- 计算信息准则(通过信息准则曲线的肘点来确定超参数n_components)
    
    ```python
    GM.bic(x) #对模型复杂度惩罚更重
    GM.aic(x) #平衡模型拟合优度和复杂度
    ```
    

## BayesianGaussianMixture

- GaussianMixture的贝叶斯变体, 使用狄利克雷过程，自动确定最佳簇数量
- 导入BayesianGaussianMixture语句
    
    ```python
    from sklearn.mixture import BayesianGaussianMixture
    ```
    
- BayesianGaussianMixture参数详解
    
    ```python
    BayesianGaussianMixture(
         n_components=1, #聚合簇参数
         covariance_type='full', #簇形状约束(full:每个簇任意方向和形状; tied:每个簇形状相同; diag:簇的坐标轴与全局坐标轴平行; spherical: 簇形状为圆形)
         tol=0.001, #收敛阈值
         reg_covar=1e-6, #协方差对角线的正则化项, 防止协方差矩阵奇异（不可逆），确保数值稳定性
         max_iter=100, #最大迭代次数(即使未达到收敛阈值，最多迭代这么多次)
         n_init=1, #算法重复初始化计算次数(防止陷入局部最优，选择最佳初始化)
         init_params='kmeans', #初始化方法
         weight_concentration_prior_type='dirichlet_process', #使用狄利克雷过程，自动确定簇数量; "dirichlet_distribution"：使用狄利克雷分布
         weight_concentration_prior=1/n_components, #控制组件的稀疏性, 值越小，越倾向于使用更少的簇数量
         mean_precision_prior=None, #均值精度先验
         mean_prior=None, #均值先验
         degrees_of_freedom_prior=None, #自由度先验
         covariance_prior=None, #协方差先验
         random_state=42,
         warm_start=False, #热启动
         verbose=0, #日志输出
         verbose_interval=10 #日志输出间隔迭代数
    )
    ```
    
- 拟合训练数据
    
    ```python
    BGM.fit(x)
    ```
    
- 获取簇数量
    
    ```python
    np.sum(BGM.weights_ > 0.01)
    ```
    