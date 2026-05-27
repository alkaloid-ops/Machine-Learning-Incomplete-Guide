

# 降维

## 主成分分析PCA

- 算法原理:
    1. 目标是找出最重要(方差最大)的n个特征数据
    2. 计算每个特征数据的方差, 从大到小依次排序, 方差越大说明特征的重要性高(方差大的特征其标签变化大, 方差小的特征其标签变化不大倾向于清一色)
    3. 去掉均值, 数据中心化
    4. 找到第一条线, 使数据投影在这条线上的方差较大(即线上的点分布较分散)
    5. 找到第二条线, 必须与第一条线垂直, 同时数据投影在第二条线方差较大
    6. 找到第三条线, 必须与第一条和第二条线相互垂直, 同时数据投影在第三条线方差较大, 依次类推
    7. 只有一条线表示高维数据降维成一维, 两条线表示降维成二维, 三条线表示降维成三维, 以此类推
- 数学计算方式(计算前务必将高维特征矩阵中心化)
    1. 对于高维特征矩阵, 计算其协方差矩阵或将高维矩阵SVD奇异值分解后取右奇异矩阵
        1. 协方差矩阵是对称矩阵, 它的特征向量是相互正交的, 即矩阵内部的任意两组向量内积为0(v_1^Tv_2=0)
        
        $$
        \Sigma = \frac{1}{n-1} X^TX
        $$
        
    2. 计算协方差矩阵的特征值和特征向量, 之后从高到低排序, 保留几个特征值与其对应的特征向量就是降维至多少维
    3. 最后用原始特征矩阵内积特征向量得到的新矩阵就是降维后的新特征矩阵Z(其中n是保留的特征向量的数量)(或取右奇异矩阵的转置的前n列, n是降维后的维度数量, 和原始矩阵做内积计算, 同样得到新特征矩阵Z)
        
        $$
        Z=X@[v_1,v_2...v_n]
        $$
        
- 导入PCA的语句
    
    ```python
    from sklearn.decomposition import PCA
    ```
    
- PCA参数详解
    
    ```python
    PCA(
        n_components='降维后的维度数(整数)', #备选参数: 0-1的浮点数(可解释性方差贡献率)(需要配合svd_solver='full'); 'mle'最大似然估计(自动选择维度)
        copy=True, #创建副本
        whiten=False, #白化处理: 在降维后、输出结果前对主成分进行的归一化操作(如果开启, 使所有降维后的各个特征具有相同方差)
        svd_solver='auto', #选择奇异值分解(SVD)的计算方法(备选参数: 'full'小数据集—精确计算但计算慢; 'randomized'大数据集—随机算法近似解计算快; 'arpack'矩阵分解指定维度等于n_components参数, 要小于原始特征矩阵的行或列数)
        tol=0, #收敛阈值(输入参数为大于0的浮点数)(值越小精度越高但计算时间越长)
        iterated_power='auto', #控制随机SVD算法的迭代次数(输入参数为整数)(迭代次数越多结果越精确,但计算时间越长)
        n_oversamples=10, #随机SVD中额外采样的数量(增加此值可提高精度,但增加计算量)(实际样本数=n_components+n_oversamples)
        power_iteration_normalizer='auto', #控制随机SVD中的归一化方法(备选参数: 'LU'LU分解—大数据集使用; 'QR'QR分解—稳定要求高使用)
        random_state=42 #仅在svd_solver参数为randomized或arpack时生效
    )
    #n_components超参数可从数值区间内遍历, 并计算解释性方差贡献率来得到一个最优值
    ```
    
- 实例化并转换数据
    
    ```python
    X = PCA().fit_transform(X)
    ```
    
- 查看降维后各个特征的可解释性方差
    
    ```python
    PCA.explained_variance_
    ```
    
- 查看可解释性方差贡献率(降维后各个特征的方差占原始特征矩阵总方差的比例)
    
    ```python
    PCA.explained_variance_ratio_
    ```
    
- 计算降维后的新特征矩阵的总方差占原始特征矩阵总方差的比例(即信息保留程度)
    
    ```python
    PCA.explained_variance_ratio_.sum()
    ```
    
- 查看PCA计算过程中SVD奇异值分解的右奇异矩阵的前n行
    
    ```python
    PCA().fit(X).components_
    ```
    
- PCA降维后还原(⚠️不能100%还原, 因为有信息丢失)
    
    ```python
    PCA().inverse_transform(X)
    ```
    

## 截断奇异值分解TruncatedSVD

- 截断的奇异值分解是一种近似的矩阵分解, 用于提高计算效率, 但不能精确重构出原始矩阵
- 直接对特征矩阵进行奇异值分解(适用于对高维稀疏矩阵、TF-IDF词频矩阵降维)
- 计算前需要将数据标准化(但不中心化)
- 截断取分解后的对角矩阵的k个最大奇异值(k取决于想要降维至多少维)
- 导入TruncatedSVD语句
    
    ```python
    from sklearn.decomposition import TruncatedSVD
    ```
    
- TruncatedSVD参数详解
    
    ```python
    TruncatedSVD(
        n_components='降维目标维度', #奇异值数量
        algorithm='randomized', #arpack精确计算
        n_iter=5, #数值大精度高
        n_oversamples=10, #随机算法中的过采样参数, 值越大结果更稳定但计算时间越长
        power_iteration_normalizer='auto', #备选参数qr: QR分解数值稳定, lu: LU分解计算更快
        random_state=42,
        tol=0 #arpack算法收敛阈值
    )
    ```
    
- 拟合转换数据
    
    ```python
    X = TruncatedSVD().fit_transform(X)
    ```
    
- 查看降维后各特征的解释性方差
    
    ```python
    X.explained_variance_
    ```
    
- 查看奇异值
    
    ```python
    X.singular_values_
    ```
    
