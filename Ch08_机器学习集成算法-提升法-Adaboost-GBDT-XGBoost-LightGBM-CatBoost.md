
# 集成算法


## 提升法Boosting

- 提升法原理
    - 模型中的各个弱评估器相互关联, 串行依次构建
    - 样本有放回抽样, 特征无放回抽样
    - 最终输出: 指定模型评估标准和输出函数
    - 袋装法目标降低偏差来提升模型泛化能力
- 提升法建模通用机制
    - 先建立一个弱评估器来做初始预测, 并计算损失函数
    - 根据上一个弱评估器的结果n-1的结果, 计算损失函数
    - 根据损失相对应的调整来建立下一个弱评估器
    - 集成模型最终输出的结果受所有建立的弱评估器的影响
- 其中AdaBoost的弱评估器可以是决策回归树或决策分类树, 其他Boosting算法的弱评估器一律都是决策回归树
- Boosting的通用计算表达式
- 模型最终输出表达式(即所有弱评估器结果的加权求和)
    
$$
Final(x_i)=\sum^k_{k=1} w_i f_k(x_i)
$$
    
- 集成模型内部某一个弱评估器的计算表达式(当前为k, 上一个为k-1, alpha为学习率或迭代步长, w为权重/梯度向量)
    
$$
F_k(x_i)=F_{k-1}(x_i)+\alpha w_i f_k(x_i)
$$
    

### 自适应提升AdaBoost

- 拟合全部训练集样本建立一棵决策树, 标记出预测误差较大或错误的样本, 对这些样本赋予权重, 后一个决策树用这些带权重的样本训练, 标记出预测误差较大或错误的样本, 再对这些样本赋予权重, 其中多次预测误差较大或错误的样本赋予更高的权重, 使后面的决策树更关注这些高权重的样本(权重遵循此消彼长的机制, 即错误样本的权重增加的同时正确的样本的权重相应会降低, 所有样本总权重不变)
- 回归损失函数linear(其中分母取预测值与真实值的差值最大的值)
    
$$
L=\frac{|\hat y_i-y_i|}{max(|\hat y_i -y_i|)}
$$
    
- 回归损失函数square(其中分母取预测值与真实值的差值最大的值的平方)
    
$$
L=\frac{|\hat y_i-y_i|^2}{max(|\hat y_i -y_i|)^2}
$$
    
- 回归损失函数exponential(其中指数分母取预测值与真实值的差值最大的值)
    
$$
L=1-e^{(\frac{-|\hat y_i-y_i|}{max(|\hat y_i -y_i|)})}
$$
    
- 二分类指数损失函数
- yhat为预测标签向量(取值为-1或1); y为真实标签向量(取值为-1或1)
    
$$
L(\hat y_i,y_i)=e^{-y_i \hat y_i}
$$
    
- 多分类指数损失函数(其中k为总类别数量—去重复)
- yhat为m行k列矩阵, 其中预测最大概率值的标签为1, 其余标签取值为-1/k-1
- y为m行k列矩阵, 其中标签类别编码和特征类别编码值一致的位置为1, 其余都为-1/k-1
    
$$
L(\hat y_i ,y_i)=e^{-\frac{1}{k}y_i \hat y_i}
$$
    
- 算法计算流程
    - 初始化训练数据的权重, 每一个样本权重为1/n
    - 从现有的训练集中抽样构成训练子集Xtrain_m(抽样时,任一样本被抽中的概率为那个样本除以所有样本的权重和) (初始权重分配均衡都是1/n, 往后权重变化时更大权重的样本被抽中的概率会更大)
    - 使用训练子集Xtrain_m和标签子集ytrain_m拟合一棵树f_n, 得到预测结果为yhat_m
    - 先计算单个样本的损失, 最后计算全部样本的加权平均损失(w’_i是样本权重)
        
$$
L = \sum^n_{i=1} w'_iL(\hat y_i ,y_i)
$$
        
    - 根据最终的加权平均损失计算置信度(其中lambda为防止分母为0的常数)(置信度越接近0越好, 模型预测结果越好)
        
$$
置信度=\frac{L}{1-L+\lambda}
$$
        
    - 根据置信度更新单个样本权重(单样本新权重=单样本原权重乘以置信度的1-单样本损失的次方)
        
$$
w'_i =  w'_i\cdot e^{-w_i y_i \hat y_i}
$$
        
    - 单个评估器f_n的权重计算
        
$$
w_i = log(\frac{1}{置信度})
$$
        
    - 集成模型内评估器的计算(w_i是评估器权重)
        
$$
F_n(x_i)=F_{n-1}+\alpha w_i f_n(x_i)
$$
        
    - 迭代停止条件: 当集成模型预测值小于等于真实值的那些评估器的权重和大于等于等于二分之一的所有评估器的权重和时, 迭代停止
        
$$
(\sum^n_{i=预测值\leq真实值} 单个评估器权重)\geq(\frac{1}{2}\sum^n_{i=1} 单个评估器权重)
$$
        
- 导入AdaBoost语句
    
    ```python
    from sklearn.ensemble import AdaBoostRegressor as ABR #回归
    from sklearn.ensemble import AdaBoostClassifier as ABC #分类
    ```
    
- AdaBoost参数详解
    
    ```python
    ABR(
        estimator=DecisionTreeRegressor(max_depth=3), #回归算法中基评估器是默认深度为3的决策回归树, 可自定义实例化基评估器
        n_estimators='评估器数量',
        learning_rate=1, #学习率
        loss='linear',#损失函数, 备选参数'square', 'exponential'
        random_state=42
    )
    
    ABC(
        estimator=DecisionTreeClassifier(max_depth=1), #分类算法中基评估器是默认深度为1的决策分类树, 可自定义实例化基评估器
        n_estimators='评估器数量',
        learning_rate=1, #学习率
        algorithm='SAMME.R', #SAMME输出具体分类结果, SAMME.R输出概率值
        random_state=42
    )
    ```
    
- 拟合训练数据
    
    ```python
    ABR().fit(Xtrain, ytrain)
    ABC().fit(Xtrain, ytrain)
    ```
    

### 梯度提升决策树GBDT

- 与随机森林并行集成决策树不同, GBDT串行集成决策树
- 同时随机森林和AdaBoost集成模型任务类别与基评估器任务类别一致, 但GBDT基评估器一律都是回归(即使是分类任务)
- 与AdaBoost不同, GBDT不修改样本权重, 同时GBDT的拟合训练目标(ytrain)是残差(残差即误差, 是预测值与真实值的误差), 根据这个残差来影响后续的弱评估器(拟合残差来建树)
- 对于多分类任务, 集成模型内的弱评估器将输出k个回归预测向量, 最终分类结果在回归预测向量的基础上套上Sigmoid二分类或Softmax多分类来输出分类结果或概率值
- GBDT二分类交叉熵损失函数(⚠️GBDT直接输出是回归预测, 在输入损失函数前需要输入Sigmoid函数转换成概率值)
    
$$
L(\hat y_i, y_i) = -\sum^n_{i=1}[y_ilog(\hat y_i)+(1-y_i)log((1-\hat y_i))]
$$
    
- GBDT多分类交叉熵损失函数(⚠️GBDT直接输出是回归预测, 在输入损失函数前需要输入Softmax函数转换成概率值)
- (其中k为分类类别数量, y_k为m行k列的稀疏矩阵, 每个样本所对应的真实标签下为1, 其余为0, yhat_k为Softmax输出的概率矩阵)
    
$$
L(\hat y_i, y_i) = -\sum^k_{i=1}y_k log(\hat y_k)
$$
    
- GBDT二分类指数损失函数
    
$$
L(\hat y_i, y_i) = e^{-y_i \hat  y_i}
$$
    
- GBDT多分类指数损失函数
    
$$
L(\hat y_i ,y_i)=e^{-\frac{1}{k}y_i \hat y_i}
$$
    
- GBDT 平方损失函数
    
$$
L(\hat y_i ,y_i)=\sum^n_{i=1} (\hat y_i -y_i)^2
$$
    
- GBDT 绝对损失函数
    
$$
L(\hat y_i ,y_i)=\sum^n_{i=1} |\hat y_i -y_i|^2
$$
    
- GBDT Huber损失函数(其中alpha为超参数, 默认值0.9)
    
$$
L_\alpha(\hat y_i,y_i)=\frac{1}{2}(\hat y_i -y_i)^2\quad if|\hat y_i-y_i|\leq \alpha \quad\quad L_\alpha(\hat y_i,y_i)=\alpha(|\hat y_i -y_i|-\frac{1}{2}\alpha)\quad\quad if|\hat y_i-y_i|> \alpha
$$
    
- GBDT Quantile损失函数(其中alpha为超参数, 默认值0.9)
    
$$
L_\alpha(\hat y_i,y_i)=\alpha|\hat y_i -y_i|\quad if: y_i\geq\hat y_i \quad\quad L_\alpha(\hat y_i,y_i)=(1-\alpha)|\hat y_i -y_i|\quad\quad if: y_i<\hat y_i
$$
    
- 高度关注异常值时, 选平方损失函数; 想要减少异常值对模型的干扰影响时, 选绝对损失函数; 平衡两者选Huber或Quantile
- 集成学习中的决策树不使用gini和entropy来衡量信息不纯度, 而使用弗里德曼均方误差(其中N_L和N_R为左右子节点的样本数, ybar_L和ybar_R为左右子节点的样本均值)(追求值越大越好)
- 弗里德曼决策树分裂理念: 左右分叉时样本数量尽量一致, 但是信息不纯度左右两边差异尽可能大, 这样能够降低模型结构复杂度, 并加速分叉
    
$$
FriedmanMSE = \frac{N_L\cdot N_R}{N_L+N_R}(\bar y_L-\bar y_R)^2
$$
    
- 训练早停
    - 当模型效果足够好, 继续迭代提升很微小时, 应该停止训练
    - 当模型效果不够好, 继续迭代反而更差时, 应该停止训练
    - 当模型效果不够好, 继续迭代可持续提升模型效果, 但需要更多的计算时间时和收敛很慢时, 应该停止训练并调整参数
- 算法计算流程
    - 初始化迭代起点f_0(求解常数C, 使得损失函数中每个真实值与常数C的误差和最小(对C的一阶偏导数为0时), 最小的误差和作为初始化迭代的起点数)
      
$$
F_0(x_i)=\arg\min_{C}\sum_{i=1}^{n}l(y_i,C)
$$
        
    - 从训练样本中抽样, 构成训练集子集
    - 对于任意一个样本计算伪残差(对上一个评估器的预测结果的损失函数中对预测值求一阶导数)
        
$$
r = -\frac{\delta l(y_i,F_{k-1}(x_i))}{\delta F_{k-1}(x_i)}
$$
        
    - 创建新的评估器, 拟合训练数据和预测标签(预测标签为伪残差/负梯度)
    - 更新迭代公式为
        
$$
F_k(x_i)=F_{k-1}(x_i)+\alpha f_k(x_i)
$$
        
    - 迭代完成最终模型输出结果为
        
$$
Final(x_i)=\sum^k_{k=1}f_k(x_i)
$$
        
- 泰勒公式展开逼近损失函数
    
$$
l(y_i,\hat y_i)=l(y_i, F_{k-1}(x_i)+ f_k(x_i))\approx l(F_{k-1}(x_i))+\frac{\delta l(F_{k-1}(x_i))}{\delta F_{k-1}(x_i)}f_k(x_i)
$$
    
- 导入GBDT语句
    
    ```python
    from sklearn.ensemble import GradientBoostingRegressor #回归
    from sklearn.ensemble import GradientBoostingClassifier #分类
    ```
    
- GBDT参数详解
    
    ```python
    GradientBoostingRegressor(
        loss='squared_error', #损失函数, 备选参数'absolute_error'绝对值损失'quantile'分位数(预测区间用)'huber'平方+绝对混合(异常值多)
        learning_rate=0.1, #学习率(树多1000低学习0.01-0.05, 树少1000高学习0.1-0.3)
        n_estimators=100, #迭代次数
        subsample=1, #抽样比例, 1为使用全部数据集样本, 当输入不为1且大于0的浮点数时, 会产生和Bagging算法一样的袋外数据(可用作验证集)(适合大数据集)
        criterion='friedman_mse', #分裂准则, 备选参数'mse'均方误差'mae'绝对误差
        init=None, #初始预测(可选评估器决策树、SVM、逻辑回归等能有fit和predict_prob功能的模型), 默认None用DummyEstimator作为预测结果
        alpha=0.9, #损失参数, 对quantile和huber生效
        validation_fraction=0.1, #验证集比例
        n_iter_no_change=5, #迭代几次后验证集损失仍无下降
        tol=0.001 #损失函数下降阈值
    
    #以下参数含义与决策树/随机森林一致
        max_features=
        max_depth=
        max_leaf_nodes=
        min_impurity_decrease=
        min_samples_split=
        min_samples_leaf=
        min_weight_fraction_leaf=
        ccp_alpha=
        random_state=
        verbose=
        warm_start=
    )
    
    GradientBoostingClassifier(
        loss='log_loss' #log_loss包含二分类和多分类的交叉熵损失函数, 备选参数exponential指数损失
      
    #以下参数含义与GBDT回归一致
        learning_rate=
        n_estimators=
        subsample=
        criterion=
        tol=
        init=
        validation_fraction=
        n_iter_no_change=
        min_samples_split=
        min_samples_leaf=
        min_impurity_decrease=
        min_weight_fraction_leaf=
        max_depth=
        max_features=
        max_leaf_nodes=
        ccp_alpha=
        random_state=
        verbose=
        warm_start=
    )
    ```
    
- 拟合训练数据
    
    ```python
    GBR = GradientBoostingRegressor().fit(Xtrain, ytrain)
    GBC = GradientBoostingClassifier().fit(Xtrain, ytrain)
    ```
    
- 查看模型实际迭代次数(早停时)
    
    ```python
    GBR.n_estimators_
    GBC.n_estimators_
    ```
    
- 查看模型袋外数据验证结果
    
    ```python
    GBR.oob_improvement_
    GBC.oob_improvement_
    #返回值为每次迭代损失函数的变化数值
    ```
    
- 查看模型训练数据结果
    
    ```python
    GBR.train_score_
    GBC.train_score_
    #返回值为每次迭代损失函数的变化数值
    ```
    

### 极端梯度提升XGBoost

- 模型最终输出表达式(所有评估器的求和, 无加权)
    
$$
Final(x)=\sum^n_{k=1}f_k(x)
$$
    
- 目标函数表达式(其中第一项为损失函数之和, 第二项控制模型结构复杂度)
    
$$
\text{Obj} = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \Omega(f_t)
$$
    
- 目标函数第一项中预测值函数展开(预测值为前面k-1棵树的预测值的总和+当前第k棵树的预测值)
    
$$
F_k(x_i)=F_{k-1}(x_i)+\eta f_k(x_i)
$$
    
- 目标函数第二项展开(这部分控制模型复杂度, gamma为叶子节点数的惩罚系数(超参数), T为叶子节点总数量, lambda或alpha为正则化系数(超参数), w为叶子权重(即每一颗树的预测值)(L2范数))

$$
\Omega(f_k)= \gamma T + \frac{1}{2}\lambda ||w_i||^2\quad 或 \quad\Omega(f_k)= \gamma T + \alpha \sum^n_{i=1}|w_i|
$$

- 单个叶子节点的结构分数公式(其中分子为单个叶子节点所有样本的损失函数的对预测值求一阶导数之和的平方, 分母为单个叶子节点的所有样本的损失函数的对预测值求二阶导数之和)
- L1正则化系数alpha加在分子上, L2正则化系数lambda加在分母上
    
$$
Score_{L1}=\frac{(\sum^n_{i=1} f'(x_i))^2+\alpha}{\sum^n_{i=1} f''(x_i)}\quad 或\quad Score_{L2}=\frac{(\sum^n_{i=1} f'(x_i))^2}{\sum^n_{i=1} f''(x_i)+\lambda}
$$
    
- 算法计算流程
    - 初始化迭代起点f_0(求解常数C, 使得损失函数中每个真实值与常数C的误差和最小(对C的一阶偏导数为0时), 最小的误差和作为初始化迭代的起点数)

$$
F_0(x_i)=\arg\min_{C}\sum_{i=1}^{n}l(y_i,C)
$$
        
    - 从训练样本中抽样, 构成训练集子集
    - 对目标函数第一部分应用泰勒公式公式展开近似(计算函数的0阶(函数本身)、1阶和2阶导数)(第一项0阶可以忽略, 第二项1阶导数g_i, 第三项2阶导数h_i)(泰勒近似后, 第一项为常数, 即上一次迭代输出的值)
        
$$
l(y_i, \hat{y}_i)\approx l(y_i, \hat{y}_i)+\frac{\delta L}{\delta \hat y_i}f_t (x_i) +\frac{1}{2}\frac{\delta^2 L}{\delta^2 \hat y_i}(f_t (x_i) )^2\approx l(f_{k-1}(x_i))+g_if_t (x_i)+\frac{1}{2}h_i(f_t (x_i) )^2
$$
        
    - 对于任意一个样本计算伪残差(对上一个评估器的预测结果的损失函数中对预测值求一阶导数作为分子，二阶导数作为分母)
        
$$
r=-\frac{\frac{\delta l(y_i, F_{k-1}(x_i))}{\delta F_{k-1}(x_i)}}{\frac{\delta^2 l(y_i, F_{k-1}(x_i))}{\delta^2 F_{k-1}(x_i)}}
$$
        
    - 创建新的评估器, 拟合训练数据和预测标签(预测标签为伪残差/负梯度)
    - 更新迭代公式为
        
$$
F_k(x_i)=F_{k-1}(x_i)+\alpha f_k(x_i)
$$
        
    - 迭代完成最终模型输出结果为
        
$$
Final(x_i)=\sum^k_{k=1}f_k(x_i)
$$
        
- 导入XGBoost语句
    
    ```python
    import xgboost as XGB
    ```
    
- 转换XGBoost专用训练集和测试集
    
    ```python
    dtrain = XGB.DMatrix(Xtrain, ytrain)
    dtest = XGB.DMatrix(Xtest, ytest)
    ```
    
- 构建XGBoost参数字典
    
    ```python
    param = {
        'booster':'gbtree', #gbtree基于CART的决策树模型, gblinear用XGBoost集成线性回归模型, dart Dropout丢弃部分树
        'rate_drop':0.2, #树丢弃比例, 当booster: dart时使用
        
        'eta': 0.1, #学习率
        'gamma': 0, #分裂所需最小增益(降低过拟合)取值0到正无穷
        'lambda':1, #L2正则化系数, 当booster是gblinear时L2正则化系数为0
        'alpha':0, #L1正则化系数
        
        'max_depth': 6, #树的最大深度
        'min_child_weight': 1, #叶子节点样本权重和的最小值
        'max_delta_step':0, #每棵树权重变化的最大步长(取1-10)(用于分类样本不均衡)
        
        'subsample':1, #样本采样比例(抽样无放回)
        'colsample_bytree':1, #特征采样比例(抽样无放回)
        'scale_pos_weight':1, #正样本权重(分类样本不均衡时)(=负样本数/正样本数)
        
        'objective': '目标损失函数', 
        'seed':42, #随机数种子
        'num_class':3, #多分类类别数(分类任务时用)
        'sample_type':'uniform', #均匀抽样无权重来丢弃树
        'normalize_type': 'tree' #增加新树时赋予新树的权重, tree表示新树的权重是所有丢弃树的权重均值, forest表示新树的权重是所有丢弃树的权重之和
        'nthread': -1, #最大线程数
    }
    #objective目标函数参数:
    	#回归: reg:squarederror平方损失, reg:squaredlogerror平方对数损失(不常用)
    	#二分类: binary:logistic二分类交叉熵损失(输出类别1的概率, 类别0需要自行计算), binary:logitrow二分类交叉熵损失(输出执行Sigmoid函数前的值), binary:hinge合页损失, 
    	#多分类: multi:softmax多分类交叉熵损失(输出各类别标签), multi:softprob多分类交叉熵(输出各类别概率)
    ```
    
- 拟合训练数据
    
    ```python
    XGB.train(
        params='参数字典',
        dtrain=dtrain,
        num_boost_round='迭代次数',
        obj='自定义目标函数', #覆盖参数字典的obj
        maximize='是否最大化评估指标', #如AUC需设为True,RMSE需设为False
        early_stopping_rounds=10, #早停机制,若验证集性能连续N轮未提升,则停止训练
        evals= #训练时的用的评估指标(通常与obj目标函数保持一致)(也可自定义其他损失函数)
        evals_result= #评估指标结果
        verbose_eval= #评估指标计算过程
        xgb_model='模型热启动',
        callbacks='回调函数列表'
    )
    ```
    
- 模型输出预测结果
    
    ```python
    XGB.predict(dtest) #回归直接返回预测值, 二分类返回概率, 多分类可返回概率或标签
    ```
    
- XGBoost交叉验证
    
    ```python
    XGB.cv(
        params='参数字典',
        dtrain='训练数据',
        num_boost_round='迭代次数',
        nfold=5, #交叉验证折数
        stratified=True, #是否分层抽样(分类问题保持类别比例,回归问题自动忽略)
        folds='自定义交叉验证', #调用sklearn的(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))
        metrics='评估指标', #如('rmse','mae','logloss','auc')元组形式
        obj='自定义目标函数', #覆盖参数字典的obj
        maximize='是否最大化评估指标', #如AUC需设为True,RMSE需设为False
        early_stopping_rounds=10, #早停机制,若验证集性能连续N轮未提升,则停止训练
        as_pandas=True, #返回结果格式是否是pandas的df, False返回字典
        verbose_eval=True, #输出过程
        show_stdv=True, #是否在结果中显示标准差来反映各折稳定性
        seed=42,
        fpreproc='数据预处理函数',
        callbacks='回调函数列表'
    )
    ```
    
- 查看早停时的最优树数量
    
    ```python
    XGB.best_ntree_limit
    ```
    
- 查看早停时的最优迭代轮次
    
    ```python
    XGB.best_iteration
    ```
    
- 查看最优迭代轮次对应的评估指标分数
    
    ```python
    XGB.best_score
    ```
    
- 查看模型初始预测值
    
    ```python
    XGB.base_score
    ```
    
- 查看特征重要性
    
    ```python
    XGB.feature_importances_
    ```
    
- 查看验证集在各轮迭代的评估指标分数
    
    ```python
    XGB.evals_result()
    ```
    

### 轻量梯度提升树LightGBM

- 初始化残差(和GBDT一样)
- 构造直方图(即分箱)
- 先将特征矩阵中的连续变量离散化(默认分位数分箱, 最多256箱), 离散化后特征矩阵所有特征就都是离散型特征了
- 使用GOSS方法选择部分样本(使用残差较大的全部样本和一部份残差较小的样本(随机抽样), 并对抽样后的样本校正权重, 使得再次迭代使用的样本总权重仍为1, 再标准化)(其中残差大的样本是重点需要通过迭代来降低误差的样本)
- 然后对特征矩阵降维(EFB方法计算冲突比例矩阵)
- EFB计算方法: 计算两个特征全为0的样本数量占全部样本数量的比例, 构建与特征数量一样的方矩阵(这个矩阵类似相关系数矩阵)
- 设定冲突阈值max_conflict_rate, 将小于这个冲突值并从最小冲突值的的两个特征开始合并
- 合并特征时第一个特征数值不变, 第二个特征数值+offset超参数值, 最后合并计算为第二个特征数值+offset值+第一个特征数值
- 同时合并多个特征时, 从最后一个特征+offset值+前一个特征+offset值+前一个特征+offset值, 一直加到第一个
- 构建树评估器, 拟合训练数据, 分裂时选择能最大限度降低损失的的节点
- 导入LightGBM语句
    
    ```python
    import lightgbm
    ```
    
- 转换专用数据集格式
    
    ```python
    train_data = lightgbm.Dataset(X_train, label=y_train)
    test_data = lightgbm.Dataset(X_test, label=y_test, reference=train_data)
    ```
    
- 构建参数字典
    
    ```python
    params = {
        # 核心参数
        'boosting_type': 'gbdt', #备选参数dart(丢弃树)
        'objective': '任务类型', #备选参数regression回归, binary二分类, multiclass多分类
        'num_iterations': 100, #树数量
        'learning_rate': 0.05, #学习率
        'num_leaves': 30, #单颗树最大叶子节点数量
        'num_threads': 0, #计算线程数
        'seed': 42,
        
        # 学习控制
        'max_depth': 7, #分裂最大深度(-1为无限制)
        'min_data_in_leaf': 50, #叶子节点最小样本数
        'min_sum_hessian_in_leaf': 0.001, #叶子节点最小Hessian和(类似min_child_weight)
        'feature_fraction': 0.8, #特征采样比例
        'bagging_fraction': 0.9, #样本采样比例
        'bagging_freq': 5, #采样频率
        'lambda_l1': 0.1, #l1正则化系数
        'lambda_l2': 0.2, #l2正则化系数
        
        # 目标参数
        'is_unbalance': True, #自动平衡正负样本权重(二分类)
        'scale_pos_weight': 1, #正样本权重乘数(>1提高召回)
        'num_class': 3, #多分类类别数量(多分类时必要参数)
        
        # 度量参数
        'metric': ['auc', 'binary_logloss', 'multi_logloss', 'rmse'], #评估指标
        
        # IO 参数
        'max_bin': 256, #最大分箱数量
        'categorical_feature': ['cat1', 'cat2']  # 类别特征名
    }
    ```
    
- LightGBM训练时参数详解
    
    ```python
    model = lightgbm.train(
        params='参数字典',
        train_set='训练集',
        num_boost_round=100, #迭代次数
        valid_sets=['验证集'], #使用测试集会有数据泄漏风险, 需从训练集单独划分一个验证集
        valid_names=['多个验证集名称'],
        feval=None, #自定义评估函数
        init_model=None, #热启动
        keep_training_booster=False, #是否保留完整的训练状态, True可继续训练, False删除部分中间状态以节省内存
        callbacks=['回调函数列表'] #需要和验证集一起使用
    )
    
    #回调函数列表: 
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),  # 早停
        lgb.log_evaluation(period=10),           # 定期输出日志
        lgb.record_evaluation(eval_result)       # 记录评估结果
    ]
    ```
    
- 模型预测测试集数据
    
    ```python
    model.predict(Xtest) #回归返回预测值, 分类返回概率值
    ```
    
- LightGBM交叉验证
    
    ```python
    lightgbm.cv(
        params='参数字典',
        train_set='训练集',
        num_boost_round=100, #迭代次数
        folds='自定义交叉验证划分方案',
        nfold=5, #交叉验证折数, folds=None时生效
        stratified=True, #是否分层抽样(回归任务不需要)
        shuffle=True, #数据洗牌
        metrics='评估指标', #这里输入会覆盖参数字典中的评估参数
        feval=None, #自定义评估函数
        init_model=False, #热启动
        fpreproc='数据预处理函数',
        callbacks='回调函数列表',
        seed=42,
        eval_train_metric=False, #开启后会额外返回训练集评估结果
        return_cvbooster=False #是否返回所有折的模型
    )
    ```
    
- 查看模型实际使用特征数量
    
    ```python
    model.num_feature()
    ```
    
- 查看早停时的最优迭代轮次
    
    ```python
    model.best_iteration
    ```
    
- 查看特征重要性
    
    ```python
    model.feature_importances_
    ```
    

### 离散提升树CatBoost

- 初始化残差(和GBDT一样)
- 对训练样本随机排序, 预测排序后的所有训练样本生成预测值, 并与真实值计算损失与梯度(这些梯度作为后续构建树评估器的ytrain)
- 然后对所有类别特征使用序列编码(CatBoost会多次重复排序样本进行目标编码, 并构建多组模型再做集成以减少方差)(每随机排序一次为一组)
- 构建对称的树评估器, 拟合编码好的样本和梯度ytrain
- 使用CatBoost建模, 不需要在特征工程环节的时候将类别特征编码为数值
- 导入CatBoost语句
    
    ```python
    from catboost import CatBoostRegressor #回归
    from catboost import CatBoostClassifier #分类
    from catboost import Pool #数据容器
    from catboost import cv #交叉验证
    ```
    
- 转换专用数据集格式
    
    ```python
    train_pool = Pool(X_train, y_train, cat_features=None)
    test_pool = Pool(X_test, y_test, cat_features=None)
    #cat_features类别标签名(自动编码无需提前独热编码或序列编码)
    ```
    
- CatBoost参数详解
    
    ```python
    CatBoostRegressor/CatBoostClassifier(
        iterations=100, #迭代次数
        learning_rate=0.03, #学习率
        depth=6, #最大深度
        loss_function='RMSE', #损失函数, 回归默认RMSE, 二分类默认Logloss, 多分类默认MultiClass/MultiClassOneVsAll
        eval_metric='RMSE', #评估指标, 回归RMSE/R2, 分类Accuracy/AUC/F1/Precision
        random_seed=42,
        verbose=0, #输出过程
    
        early_stopping_rounds=None, #早停轮数
        use_best_model=False, #是否使用验证集最佳模型
        od_type='Iter', #过拟合检测类型, Iter迭代, IncToDec指标波动
        od_wait=10, #过拟合检测等待轮数
    
        l2_leaf_reg=3, #l2正则化系数
        border_count=254, #数值特征分箱数(最大255)
        random_strength=1, #分裂随机性(进一步抑制过拟合)
        grow_policy='Lossguide', #生长策略, Lossguide损失导向, Depthwise深度优先
        min_data_in_leaf=1, #叶子节点最小样本数
        max_leaves=31, #最大叶子数(当grow_policy='Lossguide'生效)
        rsm=1, #特征采样比例
        subsample=1 #样本采样比例
    
        cat_features=None, #指定类别特征列索引/列名
        one_hot_max_size=2, #当类别唯一值≤该阈值时, 自动做One-Hot独热编码
        bootstrap_type='MVS', #样本采样方式, Bayesian贝叶斯, Bernoulli伯努利, Poisson柏松
        bagging_temperature=1, #贝叶斯采样强度(当bootstrap_type='Bayesian'生效)
        task_type='CPU', #计算设备, 备选'GPU'
        thread_count=-1, #计算线程数
        nan_mode='Min', #缺失值处理(最大值或最小值填充)
        fold_len_multiplier=2 #时间序列交叉验证的折叠长度乘数
        
        #以下参数分类专用
        class_weights=None, #类别不平衡的权重字典
        auto_class_weights='Balanced' #自动类别平衡
    )
    ```
    
- 拟合训练数据
    
    ```python
    model.fit(train_pool)
    ```
    
- 模型预测测试集数据
    
    ```python
    model.predict(Xtest) #回归输出预测值, 分类输出类别标签
    model.predict_proba(X_test) #分类输出概率值
    ```
    
- CatBoost交叉验证
    
    ```python
    cv(
        pool=
        params='参数字典',
        iterations=100, #最大迭代次数
        fold_count=5, #交叉验证折数
        folds=None, #自定义交叉验证划分策略
        shuffle=True, #数据洗牌
        seed=42,
        stratified=True, #分层采样(回归任务不需要)
        as_pandas=True, #结果输出为df, 否则为字典
        verbose=0, #输出过程
    )
    ```
    
- 查看特征重要性
    
    ```python
    model.feature_importances_
    ```
    
- 查看早停时的最优迭代轮次
    
    ```python
    model.best_iteration_
    ```
    
- 查看使用的特征名称(类别编码后名称会变化)
    
    ```python
    model.feature_names_
    ```
    
- 查看模型内部真正生效的完整参数
    
    ```python
    model.params
    ```
    
