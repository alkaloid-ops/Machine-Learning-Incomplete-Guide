

# 关联规则学习

## 非结构化事件数据预处理

- 将非结构化事件数据转换为二进制特征矩阵(矩阵中的值为布尔值)
- 导入非结构化事件数据处理语句
    
    ```python
    from mlxtend.preprocessing import TransactionEncoder
    ```
    
- 非结构化事件数据处理拟合转换数据
    
    ```python
    TE = TransactionEncoder()
    X_TE = TE.fit(X).transform(X)
    ```
    
- 逆向还原
    
    ```python
    X_INV = TE.inverse_transform(X_TE)
    ```
    
- 获取特征名
    
    ```python
    TE.columns_
    ```
    

## 支持度、置信度、提升度

- 支持度: 几个关联的数据样本在全部数据样本出现的比例
    
    $$
    Support(X,Y)=\frac{count(X,Y)}{Total}
    $$
    
- 置信度: 包含某一的类别的情况, 再满足另一类别的概率(即条件概率)
    
    $$
    Confidence(X,Y)=\frac{Support(X,Y)}{Support(X)}
    $$
    
- 提升度: 当提升度为1时, 表示X的出现与Y的出现没有什么影响(X和Y为独立事件), 当提升度大于1时, 表示X的出现提高了Y的出现概率(X和Y是正相关的), 当提升度小于1时, 表示X的出现降低了Y的出现概率(X和Y是负相关的)
    
    $$
    Lift(X,Y)=\frac{Confidence(X,Y)}{Support(Y)}
    $$
    

## 关联规则Apriori算法流程

- 目标: 找出满足用户指定的最小支持度和最小置信度的阈值的关联规则
- 假设交易数据集
    - T1: {牛奶, 面包}
    - T2: {面包, 尿布, 啤酒, 鸡蛋}
    - T3: {牛奶, 尿布, 啤酒, 可乐}
    - T4: {面包, 牛奶, 尿布, 啤酒}
    - T5: {面包, 牛奶, 尿布, 可乐}
- 设定最小支持度和最小置信度(两个超参数)(假设最小支持度为0.6; 最小置信度为0.8)
- 计算交易数据中所有类别的支持度(统计某一类别的样本数量占全部样本数据的比例)
    
    $$
    牛奶=\frac{4}{5}=0.8\quad 面包=\frac{4}{5}=0.8\quad 尿布=\frac{4}{5}=0.8\quad 啤酒=\frac{3}{5}=0.6\quad 鸡蛋=\frac{1}{5}=0.2\quad 可乐=\frac{2}{5}=0.4
    $$
    
- 筛选出支持度大于设定阈值的类别(牛奶、面包、尿布、啤酒)
- 生成两两组合的样本(牛奶与面包、牛奶与尿布、牛奶与啤酒、面包与尿布、面包与啤酒、尿布与啤酒)
- 检查组合后的样本里的类别是否都包含组合前的类别
- 计算两两组合的样本的支持度
    
    $$
    牛奶与面包=\frac{3}{5}=0.6\quad牛奶与尿布=\frac{3}{5}=0.6\quad牛奶与啤酒=\frac{2}{5}=0.4\quad面包与尿布=\frac{3}{5}=0.6\quad面包与啤酒=\frac{2}{5}=0.4\quad尿布与啤酒=\frac{3}{5}=0.6
    $$
    
- 筛选出支持度大于设定阈值的类别(牛奶与面包、牛奶与尿布、面包与尿布、尿布与啤酒)
- 生成三类别组合(要求前k-2个类别都相同)(牛奶与面包与尿布)
- 计算三类别组合的支持度
    
    $$
    牛奶与面包与尿布=\frac{2}{5}=0.4
    $$
    
- 支持度不满足最小阈值, 淘汰这个组合, 同时迭代停止(当迭代停止时, 如果没有组合筛选出, 那么最终结果为上一次迭代的组合)
- 计算最终结果的各个组合的置信度(根据设定的最小置信度筛选出组合(尿布与啤酒大于0.8))
    
    $$
    Confidence(牛奶与面包)=\frac{Support(牛奶与面包)}{Support(牛奶)}=\frac{0.6}{0.8}=0.75\quad Confidence(牛奶与面包)=\frac{Support(牛奶与面包)}{Support(面包)}=\frac{0.6}{0.8}=0.75
    $$
    
    $$
    Confidence(牛奶与尿布)=\frac{Support(牛奶与尿布)}{Support(牛奶)}=\frac{0.6}{0.8}=0.75\quad Confidence(牛奶与尿布)=\frac{Support(牛奶与尿布)}{Support(尿布)}=\frac{0.6}{0.8}=0.75
    $$
    
    $$
    Confidence(面包与尿布)=\frac{Support(面包与尿布)}{Support(面包)}=\frac{0.6}{0.8}=0.75\quad Confidence(面包与尿布)=\frac{Support(面包与尿布)}{Support(尿布)}=\frac{0.6}{0.8}=0.75
    $$
    
    $$
    Confidence(尿布与啤酒)=\frac{Support(尿布与啤酒)}{Support(尿布)}=\frac{0.6}{0.8}=0.75\quad Confidence(尿布与啤酒)=\frac{Support(尿布与啤酒)}{Support(啤酒)}=\frac{0.6}{0.6}=1
    $$
    
- 导入Apriori语句
    
    ```python
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules
    ```
    
- 计算类别和类别组合的支持度并淘汰低于阈值的类别(数据需要TransactionEncoder处理好)
    
    ```python
    frequent_items = apriori(data, min_support=0.6, use_colnames=True) #data需要以dataframe的格式
    ```
    
- 计算关联规则并选出高于阈值的组合
    
    ```python
    associate_rules = association_rules(frequent_items, metric='confidence', min_threshold=0.8)
    ```
    

## 关联规则FP-Growth算法流程

- 目标: 找出满足用户指定的最小支持度和最小置信度的阈值的关联规则
- 假设交易数据集
    - T1: {牛奶, 面包}
    - T2: {面包, 尿布, 啤酒, 鸡蛋}
    - T3: {牛奶, 尿布, 啤酒, 可乐}
    - T4: {面包, 牛奶, 尿布, 啤酒}
    - T5: {面包, 牛奶, 尿布, 可乐}
- 扫描数据集, 计算每个类别的出现次数(牛奶4、面包4、尿布4、啤酒3、鸡蛋1、可乐2)
- 设定阈值, 淘汰小于阈值的类别(假设阈值为3, 淘汰鸡蛋和可乐)
- 按照次数排降序(行间排序)
    - T1: {牛奶, 面包}
    - T3: {牛奶, 尿布, 啤酒}
    - T2: {面包, 尿布, 啤酒}
    - T4: {面包, 牛奶, 尿布, 啤酒}
    - T5: {面包, 牛奶, 尿布}
- 再将各个样本中的类别按照次数排降序(行内排序)
    - T1: {牛奶, 面包}
    - T5: {牛奶, 面包, 尿布}
    - T4: {牛奶, 面包, 尿布, 啤酒}
    - T3: {牛奶, 尿布, 啤酒}
    - T2: {面包, 尿布, 啤酒}
- 从最后一类的类别开始递归统计类别次数(不包含所选类)
- 最后一类为啤酒T4、T3、T2中, 牛奶2、面包2、尿布3, (淘汰小于阈值的牛奶和面包)未淘汰的类别与最后一类组合(尿布和啤酒)
- 倒数第二类为尿布T5、T4、T3、T2中, 牛奶3、面包3, (淘汰小于阈值的类别)未淘汰的类别与倒数第二类组合(牛奶和尿布、面包和尿布)
- 倒数第三类为面包T1、T5、T4、T2中, 牛奶3, (淘汰小于阈值的类别)未淘汰的类别与倒数第三类组合(牛奶和面包)
- 倒数第四类为牛奶T1、T5、T4、T3中, 类别为零
- 计算留下来的类别组合和组合中单个类别的支持度与置信度(设置信度阈值为0.8, 那么淘汰小于阈值的组合, 留言唯一组合尿布和啤酒)
    
    $$
    Support(尿布)=\frac{4}{5}=0.8\quad Support(啤酒)=\frac{3}{5}=0.6\quad Support(牛奶)=\frac{4}{5}=0.8\quad Support(面包)=\frac{4}{5}=0.8\quad 
    $$
    
    $$
    Support(尿布和啤酒)=\frac{3}{5}=0.6\quad Support(牛奶和尿布)=\frac{3}{5}=0.6\quad Support(面包和尿布)=\frac{3}{5}=0.6\quad Support(牛奶和面包)=\frac{3}{5}=0.6
    $$
    
    $$
    Confidence(尿布和啤酒)=\frac{Support(尿布和啤酒)}{Support(尿布)}=\frac{0.6}{0.8}=0.75\quad Confidence(尿布和啤酒) =\frac{Support(尿布和啤酒)}{Support(啤酒)}=\frac{0.6}{0.6}=1
    $$
    
    $$
    Confidence(牛奶和尿布)=\frac{Support(牛奶和尿布)}{Support(牛奶)}=\frac{0.6}{0.8}=0.75\quad Confidence(牛奶和尿布) =\frac{Support(牛奶和尿布)}{Support(尿布)}=\frac{0.6}{0.8}=0.75
    $$
    
    $$
    Confidence(面包和尿布)=\frac{Support(面包和尿布)}{Support(面包)}=\frac{0.6}{0.8}=0.75\quad Confidence(面包和尿布) =\frac{Support(面包和尿布)}{Support(尿布)}=\frac{0.6}{0.8}=0.75
    $$
    
    $$
    Confidence(牛奶和面包)=\frac{Support(牛奶和面包)}{Support(牛奶)}=\frac{0.6}{0.8}=0.75\quad Confidence(牛奶和面包) =\frac{Support(牛奶和面包)}{Support(面包)}=\frac{0.6}{0.8}=0.75
    $$
    
- 导入FP-Growth语句
    
    ```python
    from mlxtend.frequent_patterns import fpgrowth
    from mlxtend.frequent_patterns import association_rules
    ```
    
- 计算类别和类别组合的支持度并淘汰低于阈值的类别(数据需要TransactionEncoder处理好)
    
    ```python
    frequent_items = fpgrowth(data, min_support=0.6, use_colnames=True) #data需要以dataframe的格式
    ```
    
- 计算关联规则并选出高于阈值的组合
    
    ```python
    associate_rules = association_rules(frequent_items, metric='confidence', min_threshold=0.8)
    ```
    
