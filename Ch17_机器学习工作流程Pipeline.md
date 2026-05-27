
# 机器学习工作流程

## 流程打包Pipeline

- 创建标准工作流, 当fit拟合数据时, 会按照pipeline中的类顺序自动实例化来处理数据
- 导入机器学习工作流语句
    
    ```python
    from sklearn.pipeline import make_pipeline
    ```
    
- 机器学习工作流参数详解
    
    ```python
    pipe = make_pipeline(
        StandardScaler(),PCA(),LogisticRegression(),  #顺序工作流的类
        memory=None, #缓存
        verbose=True #详细日志
    )
    ```
    
- 拟合训练数据
    
    ```python
    pipe.fit(Xtrain,ytrain)
    ```
    
- 与网格搜索协同使用
    
    ```python
    #构建参数字典
    param = {
        'pca__n_components': [*range(1,11)],
        'logisticregression__penalty': ['l1','l2'],
        'logisticregression__C': [*np.linspace(0.1,1,20)],
        'logisticregression__solver': ['liblinear']
    } #参数字典中, 需要双下划线来连接模型类与模型参数, 模型使用全部小写字母, 模型参数保持原大小写
    
    #网格搜索
    GSCV = GridSearchCV(
        estimator=pipe,
        param_grid=param,
        scoring={
            'accuracy': 'accuracy',
            'recall': 'recall',
            'precision': 'precision'}, #多个指标需要字典
        refit='accuracy', #指定主要评估指标用于最佳参数组合
        cv=5,
        n_jobs=-1
    )
    ```
    

## 模型保存与加载

- 导入模型与加载保存语句
    
    ```python
    from joblib import dump, load
    ```
    
- 模型保存
    
    ```python
    dump(model, '模型文件名', compress=3) #compress参数为压缩级别, 可选0-9
    #pipeline也可以保存
    ```
    
- 模型加载
    
    ```python
    loaded_model = load('模型文件名')
    loaded_model.predict(Xtest)
    
    #对于超大模型, 使用内存映射加载（不读入内存）
    model = load('模型文件名', mmap_mode='r')
    ```
    

## 增量学习

- 对于已经拟合数据训练过的模型继续添加新数据继续训练
- 新数据务必和原训练数据的特征数量和类型以及顺序一致, 同时新数据的预处理和特征工程都要和原数据严格一致(使用Pipeline保存原数据的处理过程)
- 需要开启模型配置参数热启动(warm_start=True)(⚠️不是所有模型都有热启动参数)
- 新数据中可以保留最近的部分数据作为测试集
- 当拟合新数据后模型性能下降时, 需要再次调整参数或架构升级(克隆旧模型的参数拟合新数据训练子模型, 然后融合模型)
- 对于数据集过大难以导入时(除了使用数据库SQL相关方法)
    
    ```python
    for i in range(0,1000000,100000):
        df = pd.read_csv('file_path',skiprows=i,nrows=100000) #以10万为单位导入, 100万为假设上限, 探测数据最大行数区间
    ```
    
- 调用训练好的数据继续拟合新数据
    
    ```python
    model.set_params(n_estimators=200)  # 例如随机森林增加100棵树用于新数据, 旧的树以及参数不变
    model.fit(X_new, y_new)
    ```