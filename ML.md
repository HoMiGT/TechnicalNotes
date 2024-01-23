# 机器学习
> |Python库|Rust替代库|库地址|
> |:--:|:--:|:--:|
> |numpy|ndarray|[RustML之ndarray](https://crates.io/crates/ndarray)|
> |pandas|polars|[RustML之polars](https://crates.io/crates/polars)|
> |scikit-learn|linfa|[RustML之linfa](https://crates.io/crates/linfa)|
> |matplotlib|plotters|[RustML之plotters](https://crates.io/crates/plotters)|
> |pytorch|tch-rs|[RustML之tch-rs](https://crates.io/crates/tch)|
> |networks|petgragh|[RustML之petgraph](https://github.com/petgraph/petgraph)|
## 1. 基础概念
> * 概念：是一门通过编程让计算机从数据中进行学习的科学和艺术。
> * 通用定义：
>   * 机器学习是一个研究领域，让计算机无须进行明确编程就具备学习能力 ———— 亚瑟·萨缪尔（Arthur Samuel），1959
>   * 一个计算机程序利用经验E来学习任务T，性能是P，如果针对任务T的性能P随着经验E不断增长，则称为机器学习。 ———— 汤姆·米切尔（Tom Mitchell），1997
> 
> ```mermaid
> graph LR
> A(数据) --> B[训练机器学习算法]
> B --> C{评估解决方案}
> C --> |合格| D[发布]
> C --> |不合格| E[分析错误]
> E --> F[研究问题]
> F-->B
> G[传统方法]
> ```
> ```mermaid
> graph LR
> A(数据) --> B[训练机器学习算法]
> B --> C{评估解决方案}
> C --> |合格| D[发布]
> D --> E[更新数据]
> E --> A
> E --> B
> F[机器学习方法]
> ```
## 2. 应用示例
> * 图像分类 -- **卷积神经网络(CNN)**
>   * 分析生产线上的产品图像对产品进行自动分类
>   * 通过脑部扫描发现肿瘤
> * 自然语言处理 -- **NLP、循环神经网络(RNN)、CNN、Transformer**
>   * 自动分类新闻，文本分类
>   * 论坛中自动标记恶评
>   * 自动对长文章做总结
>   * 创建一个聊天机器人或者个人助理
> * 回归问题 -- **线性回归或多项式回归、SVM回归、随机森林回归或人工神经网络，考虑过去的性能指标可以使用RNN、CNN、Transformer**
>   * 基于很多性能指标来预测公司下一年的收入
> * 语音识别 -- **RNN、CNN或Transformer**
>   * 让应用对语音命令做出反应
> * 聚类问题
>   * 检测信用卡欺诈
>   * 基于客户的购买记录来对客户进行分类，对每一类客户设计不同的市场策略
> * 推荐系统 -- **人工神经网络**
>   * 基于以前的购买记录给客户推荐可能感兴趣的产品
> * 强化学习 -- **RL**
>   * 为游戏建造智能机器人
>
## 3. 系统分类
> * 是否在人类监督下训练
>   * 有监督学习(标签)
>     * K-近邻算法
>     * 线性回归
>     * 逻辑回归
>     * 支持向量机(SVM)
>     * 决策树和随机森林
>     * 神经网络
>   * 无监督学习(无标签) -- 好的做法：使用降维算法减少训练数据的维度，再使用有监督学习
>     * 聚类算法
>     * K-均值算法
>     * DBSCAN
>     * 分层聚类分析(HCA)
>     * 异常检测和新颖性检测
>     * 单类SVM
>     * 孤立森林
>     * 可视化和降维
>     * 主成分分析(PCA)
>     * 核主成分分析
>     * 局部线性嵌入(LLE)
>     * t-分布随机近邻嵌入(t-SNE)
>     * 关联规则学习
>     * Apriori
>     * Eclat
>   * 半监督学习(部分标签)
>     * 深度信念网络(DBN)
>   * 强化学习 -- 应用在机器人学习
> * 是否可以动态地进行增量学习
>   * 在线学习
>   * 批量学习 -- 先训练系统，再投入生产环境
> ```mermaid
> graph LR
> A[大量数据] --> B[数据切片多份数据源]
> B --> C[训练机器学习算法]
> C --> D{评估解决方案}
> D --> |合格| E[发布]
> D --> |不合格| F[分析误差]
> F --> G[研究问题]
> G --> B
> H[在线学习处理超大数据集]
> ```
> * 是简单地将新的数据点和已知的数据点进行匹配，还是像科学家那样，对训练数据进行模式检测然后建立一个预测模型
>   * 基于实例的学习 -- 先给数据
>   * 基于模型的学习 -- 先定公式
>        
> **思路**
> 1. 研究数据
> 2. 选择模型
> 3. 使用训练数据训练(即前面学习算法搜索模型参数值，从而使成本函数最小化的过程)
> 4. 最后，应用模型对新示例进行预测或推断，希望模型的泛化结果不错
## 4. 主要挑战与测试验证
> **主要挑战**
> * 训练数据的数量不足
> * 训练数据不具代表性
> * 低质量数据
> * 无关特征
> * 过拟合训练数据
> * 欠拟合训练数据    
>
> **测试验证**
> 了解模型的泛化能力的唯一办法就是让模型真实地去处理新场景。
> 更好的选择是将数据分割成俩部分：*训练集*和*测试集*
> * 超参数调整和模型选择
>   * 训练集训练，用测试集验证至最佳模型，然后再将全部数据集合训练，形成的模型就是最佳模型
>   * 犹豫模型选择，就都尝试一遍，对比哪个泛化效果更好就选哪个
> * 数据不匹配
>   * 剔除不合理的数据
>   * 模型假设，*想知道哪个模型最好的方法就是对所有模型进行评估，但实际上是不可能的，因此会对数据做出一些合理的假设，然后只评估部分合理的模型*
>  
## 5. 机器学习项目
> 1. 框出问题并看整体   
>   * 框出问题并看整体 **建立模型本身不是最终目标，而是如何使用这个模型，如何从中获益。**  **知全貌而窥一隅，窥一隅而知全貌** **不知道如何思考就提出问题，框架问题：是有监督学习、无监督学习还是强化学习？是分类任务、回归任务还是其他任务？应该使用批量学习还是在线学习技术？**
>   * 选择性能指标 *不同的类型问题，对应的指标不同，如回归问题的典型性能指标是均方根误差(RMSE)*       
> **均方根误差(RMSE)**
> $$RMSE(X,h)=\sqrt{\frac{1}{m} \sum_{i=1}^{m} (h(x^i)-y^i)^2}$$
> m是测试RMSE的数据集中的实例数     
> $x^i$ 是数据集中第i个实例所有特征值(不包含标签)的向量，而 $y^i$ 是其标签(该实例的期望输出值)    
> X是一个矩阵，其中包含数据集中所有实例的所有特征值(不包含标签),每个实例只有一行，第i行等于x的转置，记作 $(X^i)^T$      
> h是系统的预测函数，也称为假设。当给系统输入一个实例的特征向量 $x^i$ 时，它会为该实例输出一个预测值 $\hat{y} = h(x^i)$       
> **平均绝对误差(MAE)**
> $$MAE(X,h)=\frac{1}{n} \sum_{i=1}^{n} |h(x_i) - y_i|$$      
> 计算平方和的根(RMSE) 与欧几里得范数相对应，也称 $l_2$ 范数，记作 $||\cdot||_2$ 或 $$||\cdot||$$    
> 计算绝对值之和(MAE) 也称曼哈顿范数，与 $l_1$ 范数对应，记作 $||\cdot||_1$
> 范数指针越高，它越关注大值而忽略小值。这就是RMSE对异常值比MAE更敏感的原因。当离群值呈指数形式稀有时(如钟形曲线)，RMSE表现非常好，通常是首选。
>   * 检查假设    
> 列举和验证到目前为止做出的假设，是一个非常好的习惯。
> 2. 获取数据
>   * 快速查看数据结构
>   * 创建测试集：数据集的20%，完美的方案是每个实例标识符的哈希值<=最大哈希值的20%，这样可以确保测试集在多个运行里是一致的。*关键的一步*
> 3. 研究数据以获得深刻见解
>   * 将训练集的数据可视化，如果数据量庞大，可以抽样一个探索集。
>   * 寻找相关性，如线性相关，非线性相关
>   * **试验不同属性的组合**，在准备给机器学习算法输入数据之前，做的最后一件事应该是尝试各种属性的组合
> 4. 机器学习算法的数据准备
> 5. 探索许多不同的模型，并列出最佳模型
>   * 特征缩放，最小-最大缩放(归一化)，将值重新缩放使其最终范围归于0~1之间。标准化，首先减去平均值(标准化的均值总是零),然后除以方差，从而使结果的分布具备单位方差。
> 6. 微调模型，并将它们组合成一个很好的解决方案
>   * 网格搜索：手动调整超参数，直到找到一组很好的超参数组合。scikit-learn GridSearchCV 
> 7. 展示解决方案
> 8. 启动、监控和维护系统
## 6. 分类问题
> ### 1. 二元分类器
> * 随机梯度下降(SGD)分类器，有效处理非常大型的数据集，非常适合在线学习
> ### 2. 性能测量
> * cross_val_score(准确率)  K-折交叉验证法 准确率通常无法成为分类器的首要性能指标
> * cross_val_predict(混淆矩阵) K-折交叉验证法 **评估分类器性能首选** 返回是决策分数，而非预测结果
>   $$精度=\frac{TP}{TP+FP}$$
>   其中 TP是真正类的数量，FP是假正类的数量。
>   $$召回率=\frac{TP}{TP+FN}$$
>   其中 FN是假负类的数量
>   精度通常与另一个指标一起使用，这个指标就是召回率，也称为灵敏度或者真正类率
> ```Python
> from sklearn.metrics import precision_score,recall_score
> precision_score(y_train_5,y_train_pred)  # 精度
> recall_score(y_train_5,y_train_pred)  # 召回率
> ```
> $$F_1=\frac{2}{\frac{1}{精度} + \frac{1}{召回率}} = 2 * \frac{精度*召回率}{精度+召回率} = \frac{TP}{TP+ \frac{FN+FP}{2}}$$
> 要计算出 $F_1$ 分数，只需要调用如下函数
> ```Python
> from sklearn.metrics import f1_score
> f1_score(y_train_5,y_train_pred)
> ```
> * 精度/召回率权衡，不能同时增加精度又减少召回率，反之亦然
>   提高阈值 decision_function() 该方法返回每个实例的分数，根据这些分数，使用任意阈值进行预测       
> ```Python
> from sklearn.metrics import roc_auc_score
> roc_auc_score(y_train_5,y_scores)
> ```
>   
> ### 3. 多分类器
> * sklearn.svm.SVC 支持向量机分类器
> * sklearn.multiclass.OneVsRestClassifier 一对剩余策略分类器
> ```Python
> from sklearn.svm import SVC
> from sklearn.multiclass import OneVsRestClassifier
> ovr_clf = OneVsRestClassifier(SVC())
> ovr_clf.fit(X_train,y_train)  
> ovr_clf.predict([some_digit])
> ```
> * sklearn.linear_model.SGDClassifier  SGD分类器直接可以将实例分为多个类，调用decision_function() 可以获得分类器将每个实例分类为每个类的概率列表
> ### 4. 误差分析
> 尝试多个模型，列出最佳模型并用GridSearchCV对其超参数进行微调，尽可能自动化。
> 假如已经找到一个有潜力的模型，希望找到一些方法对其进一步改进，方法之一就是分析其错误类型。
> 首先看混淆矩阵，使用cross_val_predict()函数进行预测，然后调用confusion_matrix()函数
> ```Python
> y_train_pred = cross_val_predict(sgd_clf,X_train_scaled,y_train,cv=3)
> conf_mx = confusion_matrix(y_train,y_train_pred)
> print(conf_mx)
> plt.matshow(conf_mx,cmap=plt.cm.gray)
> plt.show() 
> ```
> ### 5. 多标签分类
> * sklearn.neighbors.KNeighborsClassifier
> ```Python
> from sklearn.neighbors import KNeighborsClassifier
> y_train_large = (y_train>=7)
> y_train_odd = (y_train %2==1)
> y_multilabel = np.c_[y_train_large,y_train_odd]
>
> knn_clf = KNeighborsClassifier()
> knn_clf.fit(X_train,y_multilabel)
> knn_clf.predict([some_digit])
> ```
>   评估多标签分类
> ```Python
> y_train_knn_pred = cross_val_predict(knn_clf,X_train,y_multilabel,cv=3)
> f1_score(y_multilabel,y_train_knn_pred,average="macro")  # 根据自身权重 average="weighted" 
> ```
> ### 6. 多输出分类
> 有噪音，依然可以正常识别出分类
## 7. 训练模型
> ### 1. 线性模型
> 对输入特征加权求和，再加上偏置项(也称截距项)的常数，以此进行预测    
> 线性回归模型预测
> $\hat{y}=\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$
> 其中 $\hat{y}$ 是预测值 n是特征数量 $x_n$ 是第n个特征值 $\theta_n$ 是第n个模型参数
>
>
> 线性回归模型的MES成本函数
> $$MSE(\theta) = (X,h_\theta) = \frac{1}{n} \sum_{i=1}{n} (\theta^T x^(i) - y^(i))^2$$
>
> 
> 标准方程,为了得到使成本函数最小的 $\theta$ 值
> $$\hat{\theta}=(X^TX)^{-1} X^T y$$
> 其中 $\hat{\theta}$ 是使成本函数最小的 $\theta$ 值
> y 是包含 $y_1$ 到 $y_m$ 的目标值向量     
> ### 2. 梯度下降
> 梯度下降的中心思想就是迭代地调整参数从而使成本函数最小化
> 梯度下降一个重要参数使 **步长**，取决于超参数学习率。
> ### 3. 批量梯度下降
> 主要问题是： 训练集很大时，算法会特别慢     
> 成本函数的偏导数
> $$\frac{\partial}{\partial \theta_j}MSE(\theta) = \frac{2}{m} \sum_{i=1}^{m} (\theta^T x^{(i)} - y^{(i)}) x_j^{(i)}$$
> 对偏导数的一次性计算公式     
> $$\nabla_\theta MSE(\theta) = \frac{2}{m}X^T(X \theta-y)$$
> 梯度下降步骤
> $$\theta^{(下一步)}=\theta-\eta \nabla_\theta MSE(\theta)$$
> 其中 $\eta$ 是学习率，用梯度向量乘以 $\eta$ 确定下坡步长的大小
> ```Python
> eta = 0.1  # 学习率
> n_iterations = 1000
> m = 100
> theta = np.random.randn(2,1)
> for iteration in range(n_iterations):
>     gradients  = 2/m * X_b.T.dot(X_b.dot(theta) - y)
>     theta = theta - eta * gradients
> ```
> ### 4. 随机梯度下降
> ```Python
> n_epochs = 50
> t0, t1 = 5, 50  # 学习计划超参数
> def learning_schedule(t):
>     return t0 / (t+t1)
> theta = np.random.randn(2,1)  # 随机初始化
> for epoch in range(n_epochs):
>     for i in range(m):
>         random_index = np.random.randint(m)
>         xi = X_b[random_index:random_index+1]
>         yi = y[random_index:random_index+1]
>         gradients = 2 * xi.T.dot(xi.dot(theta)-yi)
>         eta = learning_schedule(epoch * m+i)
>         theta = theta - eta * gradients
> ```
> sklearn.linear_model.SGDRegressor
> 小批量梯度下降 
> ### 5. 多项式回归
> 当存在多个特征时，多项式回归能够找到特征之间的关系。PolynomialFeatures还可以将特征的所有组合添加到给定的多项式阶数
> ### 6. 学习曲线
> 改善过拟合模型的一种方法时向其提供更多的**训练数据**，直到验证误差达到训练误差为止。
> 偏差/方法权衡
>   统计学和机器学习的重要理论成果是以下事实：模型的泛化误差可以表示为三个非常不同的误差之和：
>     * 偏差(这部分泛化误差的原因在于错误的假设)
>     * 方差(由于模型对训练数据的细微变化过于敏感，具有许多自由度的模型)
>     * 不可避免的误差(数据本身的噪声所致，减少这部分误差的唯一方法就是清理数据)
>
> 增加模型的复杂度通常会显著提升模型的方差并减少偏差。反过来，降低模型的复杂度则会提升模型的偏差并降低方差，这就是为什么称其为权衡      
> ### 7. 正则化线性模型 (正则，就是规范行为)
> 减少过拟合的一个好的方法是对模型进行正则化(即约束模型): 拥有的自由度越少，则过拟合数据的难度就越大
> 对于线性模型，正则化通常是通过约束模型的权重来实现的
>   * 岭回归
>     岭回归(Tikhonov正则化)，是线性回归的正则化版本。     
>     在训练期间将正则化项添加到成本函数中。训练完模型后，使用非正则化的性能度量来评估模型的性能     
>     岭回归成本函数    
>     $$J(\theta)=MSE(\theta)+\alpha \frac{1}{2} \sum_{i=1}^{n} \theta_{i}^2$$
>     偏置项 $\theta_0$ 没有进行正则化。在岭回归之前缩放数据很重要，因为对输入特征的缩放敏感。大多数正则化模型都需要如此。   
>     闭式解的岭回归
>     $$\hat{\theta}=(X^T X + \alpha A)^{-1}X^Ty$$
>     ```Python
>     from sklearn.linear_model import Ridge
>     ridge_reg = Ridge(alpha=1,solver="cholesky")
>     ridge_reg.fit(X,y)
>     ridge_reg.predict([[1.5]])
>
>     sgd_reg = SGDRegressor(penalty="l2")  # 设置l2表示SGD在成本函数中添加一个正则项，等于权重向量的l2范数的平方的一半，即岭回归     
>     sgd_reg.fit(X,y.ravel())
>     sgd_reg.predict([[1.5]])
>     ```
>   * Lasso回归   
>     最小绝对收缩和选择算子回归，是增加的权重向量的l1范数，而不是l2范数的平方的一半       
>     Lasso回归成本函数      
>     $$J(\theta)=MSE(\theta)+\alpha \sum_{i=1}^{n} |\theta_i|$$
>     ```Python
>     from sklearn.linear_model import Lasso
>     lasso_reg = Lasso(alpha=0.1)
>     lasso_reg.fit(X,y)
>     lasso_reg.predict([[1.5]])
>
>     sgd_reg = SGDRegressor(penalty="l1")  # 设置l1表示 Lasso回归
>     sgd_reg.fit(X,y.ravel())
>     sgd_reg.predict([[1.5]])
>     ```
>   * 弹性网络
>     弹性网络介于岭回归和Lasso回归的中间地带。正则项是岭回归和Lasso正则项的简单混合，可以控制混合比r。当r=0时，弹性网络等效于岭回归，当r=1时，弹性网络等效于Lasso回归        
>     弹性网络的成本函数     
>     $$J(\theta)=MSE(\theta)+r \alpha \sum_{i=1}^{n} |\theta_i| + \frac{1-r}{2} \alpha \sum_{i=1}^{n} \theta_i^2$$
>     通常来讲，有正则化--哪怕很小，总比没有更可取一些。大多数情况下，该避免使用纯线性回归。岭回归是个不错的默认选择。
>     如果，实际用到的特征只有少数几个，那就应该更倾向于Lasso回归或者弹性网络，因为它们将无用特征的权重降为零。一般而言，弹性网络优于Lasso回归
>     ```Python
>     from sklearn.linear_model import ElasticNet
>     elastic_net = ElasticNet(alpha=0.1,l1_ratio=0.5)
>     elastic_net.fit(X,y)
>     elastic_net.predict([[1.5]])
>     ```
>   * 提前停止
>     对于梯度下降这一类迭代学习的算法，在验证误差达到最小值时停止训练
>     使用随机和小批量梯度下降时，曲线不是那么平滑，可能很难知道你是否到达了最小值。一种解决方案是仅在验证错误超过最小值一段时间后停止，然后回滚模型参数到验证误差最小的位置。
>     ```Python
>     from sklearn.base import clone
>     # 预处理数据
>     poly_scaler = Pipeline([
>         ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
>         ("std_scaler", StandardScaler())
>     ])
>     X_train_poly_scaled = poly_scaler.fit_transform(X_train)
>     X_val_poly_scaled = poly_scaler.transform(X_val)
>     sgd_reg = SGDRegressor(max_iter=1,tol=-np.infty,warm_start=True,penalty=None,learning_rate="constant",eta0=0.0005)  # warm_start=True 表示停止的地方继续训练，而不是从头开始。
>     minimum_val_error = float("inf")
>     best_epoch = None
>     best_model = None
>     for epoch in range(1000):
>         sgd_reg.fit(X_train_poly_scaled,y_train)
>         y_val_predict = sgd_reg.predict(X_val_poly_scaled)
>         val_error = mean_squared_error(y_val,y_val_predict)
>         if val_error < minimum_val_error:
>             minimum_val_error = val_error
>             best_epoch = epoch
>             best_model = clone(sgd_reg)
>     ```
>
> ### 8. 逻辑回归
> * 估计概率    
>   逻辑回归模型的估计概率(向量化形式)    
>   $$\hat{p}=h_\theta(x)=\sigma(x^T \theta)$$    
>   其中 $\sigma$ 是一个sigmoid函数(即S型函数)，输出一个介于0和1之间的数字        
>   逻辑函数    
>   $$\sigma(t)=\frac{1}{1+exp(-t)}$$
>   逻辑回归模型预测     
>   $$\hat{y} = \begin{cases} 0, \hat{p} < 0.5 \ 1,\hat{p} geq 0.5 \end{cases}$$
> * 训练和成本函数    
>   逻辑回归成本函数(对数损失)    
>   $$J(\theta)=- \frac{1}{m} \sum_{i=1}^{m} [y^{(i)}log(\hat{p}^{(i)}+(1-y^{(i)}))log(1-\hat{p}^{(i)})]$$
>   逻辑成本函数偏导数
>   $$\frac{\partial}{\partial \theta_j}J(\theta)=\frac{1}{m} \sum_{i=1}^{m}(\sigma(\theta^Tx^{(-1)})-y^{(i)})x_j^{(i)}$$
> * 决策边界    
>   模型估算概率的界定点，即模型的决策边界
> * Softmax回归    
>   类k的Softmax分数
>   $$S_k(x)=x^T \theta^{(\theta)}$$
>   每个类都有自己的特定参数向量 $\theta^{(\theta)}$ 所有这些向量通常都作为行存储在参数矩阵中
>   Softmax函数    
>   $$\hat{P_k}=\sigma(S(x)) = \frac{exp(S_k(x))} {\sum_{j=1}^{k} exp(S_j(x))}$$
>   其中 k 是系数 S(x) 是一个向量，其中包含实例x的每个类的分数 $\sigma(S(x))_k$ 是实例x属于类k的估计概率，给定该实例每个数的分数。
>   Softmax回归分类预测    
>   $$\hat{y}=arg_k max \sigma(S(x))_k = arg_k max S_k(x)=arg_k max((\theta^{(k)})^Tx)$$
>   交叉熵成本函数
>   $$J(\Theta)=- \frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} log(\hat{p}_k^{(i)})$$
> ## 8. 支持向量机
> SVM 特别适合用于中小型复杂数据集的分类    
> * 线性SVM分类 
>   SVM 分类器视为在类之间模拟尽可能的最宽的街道，也叫大间隔分类。
>   sklearn.svm.LinearSVC
> * 非线性SVM分类
>   多项式内核，解决非线性的，增加特征
>   ```Python
>   多项式内核
>   from sklearn.svm import SVC
>   poly_kernel_svm_clf = Pipeline([
>     ("scaler",StandardScaler()),
>     ("svm_clf",SVC(kernel="poly",degree=3,coef0=1,C=5))
>   ])
>   poly_kernel_svm_clf.fit(X,y)
>   ```
>   高斯RBF内核
>   ```Python
>   # 高斯内核 rbf
>   rbf_kernel_svm_clf = Pipeline([
>     ("scaler",StandardScaler()),
>     ("svm_clf",SVC(kernel="rbf",gamma=5,C=0.001))
>   ])
>   rbf_kernel_svm_clf.fit(X,y)
>   ```
> * SVM回归
>   ```Python
>   from sklearn.svm import LinearSVR
>   svm_reg = LinearSVR(epsilon=1.5)
>   svm_reg.fit(X,y)
>   ```
>   ```Python
>   from sklearn.svm import SVR
>   svm_poly_reg = SVR(kernel="poly",degree=2,C=100,epsilon=0.1)
>   svm_poly_reg.fit(X,y)
>   ```
>   SVR类是SVC类的回归等价物，LinearSVR类也是LinearSVC类的回归等价物。SVM也可用于异常值检测
> * 工作原理(待完善)
> ## 9. 决策树
> 决策树不稳定，相同的数据集可能训练出不同的模型，正交的决策边界
> ## 10. 集成学习和随机森林
> 集成方法将它们组合成一个更强的预测器
> * 投票分类器    
>   已经训练好了一些分类器，每个分类器的准确率约为80%，大概包括一个逻辑回归分类器，一个SVM分类器，一个随机森林分类，一个k-近邻分类器等等。    
>   标准：聚合多个分类器的预测结果，投票得出分类类别。    
>   结果：投票法分类器的准确率通常比集成中最好的分类器还要高。    
>   备注：即使每个分类器都是弱学习器(意味着它仅比随机猜测好一点)，通过集成依然可以实现一个强学习器(高准确率)，只要有足够大数量并且足够多种类的弱学习器即可。    
>   注意：当预测器尽可能互相独立时，集成方法的效果最优。获得多种分类器的方法之一就是使用不同的算法进行训练。这会增加它们犯不同类型错误的机会，从而提升集成的准确率。
>   ```Python
>   from sklearn.datasets import make_moons
>   X,y = make_moons(n_samples=100,noise=0.15)
>   split_ = int(len(y) * 0.8) 
>   X_train,y_train,X_test,y_test = X[:split_],y[:split_],X[split_:],y[split_:]
>
>   # 集成学习-投票分类器
>   # 硬投票
>   from sklearn.ensemble import RandomForestClassifier
>   from sklearn.ensemble import VotingClassifier
>   from sklearn.linear_model import LogisticRegression
>   from sklearn.svm import SVC
>   log_clf = LogisticRegression()
>   rnd_clf = RandomForestClassifier()
>   svm_clf = SVC()
>   voting_clf = VotingClassifier(
>     estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],
>     voting='hard'
>   )
>   voting_clf.fit(X_train,y_train)
>
>   from sklearn.metrics import accuracy_score
>   for clf in (log_clf,rnd_clf,svm_clf,voting_clf):
>     clf.fit(X_train,y_train)
>     y_pred  = clf.predict(X_test)
>     print(clf.__class__.__name__,accuracy_score(y_test,y_pred))
>
>
>   # 软投票
>   svm_clf = SVC(probability=True)
>   voting_clf = VotingClassifier(
>     estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],
>     voting='soft'
>   )
>   voting_clf.fit(X_train,y_train)
>
>   for clf in (log_clf,rnd_clf,svm_clf,voting_clf):
>     clf.fit(X_train,y_train)
>     y_pred  = clf.predict(X_test)
>     print(clf.__class__.__name__,accuracy_score(y_test,y_pred))
>   
>   ```  
> * bagging和pasting
>   bagging: 采样放回样本，自举汇聚法。
>   pasting：采样样本不放回
>   ```Python
>   # sklearn中的bagging和pasting 
>   from sklearn.ensemble import BaggingClassifier
>   from sklearn.tree import DecisionTreeClassifier
>   bag_clf = BaggingClassifier(
>     DecisionTreeClassifier(),n_estimators=500,
>     max_samples=80,bootstrap=True,n_jobs=-1,oob_score=True  # oob_score=True 包外评估，平均只对63%的训练实例进行采样，剩余的37%未被采样的训练实例 开启之后，会自动进行交叉验证
>   )
>   
>   bag_clf.fit(X_train,y_train)
>   y_pred = bag_clf.predict(X_test)
>   y_pred = bag_clf.predict_proba(X_test)   # 使用该方法，自动执行软投票而不是硬投票
>   ```
>   总结: bagging生成的模型通常更好，原因是自举法给每个预测器的训练子集引入了更高的多样性，所以最后bagging比pasting的偏差略高，预测器之间的关联度更低，所以集成的方法降低。
> * 随机补丁和随机子空间
>   对处理高维输入(例如图像)特别有用。对训练实例和特征都进行抽样，这称为随机补丁方法。
>   保留所有训练实例(即bootstrap=False且max_samples=1.0)但是对特征进行抽样(即bootstrap_features=True并且 max_features<1.0) 这称为随机子空间法。
>   对特征抽样给预测器带来更大的多样性，所以以略高一点的偏差换取了更低的方差。
> * 随机森林
>   随机森林是决策树的集成，通常用bagging，训练集大小通过max_samples来设置。除了先构建一个BaggingClassifier然后传入DecisionTreeClassifier，还有一种方法就是使用RandomForestClassifier。
> * 极端随机树
>   极端随机树比常规随机森林要快的很多，因为在每个节点上找到每个特征的最佳阈值是决策生长中最耗时的任务之一
>   备注：通常很难预先知道一个RandomForestClassifier是否会比一个ExtraTreesClassifier更好或更坏。唯一的方法是俩种都尝试一遍，然后使用交叉验证进行比较，还需要使用网格搜索调整超参数。
> * 
