

# K近邻

- 模型学习训练集的数据以及标签样本分布, 当有新的数据需要预测时, 指定k个离新数据最近距离的样本, 对于分类, 哪一类数量多新数据就判定为哪一类, 对于回归, 计算最近样本数据的均值作为新数据的预测结果
- 数据需要无量钢化
- 超参数K的最优值选择: 对于分类, 取奇数以避免k个样本类别数均等, 对于回归可奇数可偶数, 结合交叉验证和学习曲线来调参
- 导入KNN语句
    
    ```python
    from sklearn.neighbors import KNeighborsRegressor #回归
    from sklearn.neighbors import KNeighborsClassifier #分类
    ```
    
- KNN参数详解
    
    ```python
    KNeighborsRegressor/KNeighborsClassifier(
        n_neighbors=5, #超参数k
        weights='uniform', #权重, uniform所有邻居数据权重相等, distance距离越近权重越大
        algorithm='auto', #备选参数, ball_tree高维数据, kd_tree中小数据, brute小数据(粗糙计算)
        leaf_size=30, #叶子节点样本数, 使用ball_tree和kd_tree生效
        p=2, #曼哈顿距离1, 欧几里得距离2
        metric='minkowski', #距离度量标准
        metric_params=None, #传递给自定义距离函数的额外参数
        n_jobs=-1 #计算单元
    )
    ```
    
- 拟合训练数据
    
    ```python
    KNeighborsRegressor().fit(Xtrain, ytrain)
    KNeighborsClassifier().fit(Xtrain, ytrain)
    ```
    
