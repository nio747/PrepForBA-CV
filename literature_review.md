[TOC]

**TO DO**: 找机会整理cs231n L3, L4的笔记，补充L7的电子笔记。有时间看看内附的论文。

格式：重要内容：黑体/高亮/!!（双半角感叹号）。 引用本文章： 引用框。  引用与本文章有所关联的外部文章：斜体。 自己的review：下划线。 看不懂的标记：??（双半角问号）。待办：**TO DO**

# General: Transfer learning

2018.04.11, 3 Papers

[github资料](https://github.com/jindongwang/transferlearning#0latest) 王晋东整理。

## 入门简介

[**极简介绍**](https://github.com/jindongwang/transferlearning/blob/master/doc/%E8%BF%81%E7%A7%BB%E5%AD%A6%E4%B9%A0%E7%AE%80%E4%BB%8B.md)

已有知识：源域 source domain； 要学习的新知识：目标域 target domain；

Source task --> Knowledge --> [Learning system] <--Target task

学习方式：

- 基于样本：对源域中有标样本的加权利用  #??
- 基于特征：将源域和目标域映射到相同的空间（或其一到另一空间），最小化源域和目标域的距离。
- 基于模型：将源域和目标域的模型和样本结合，调整模型参数
- 基于关系：在源域学习概念之间的关系，类比到目标域中。

重要前提：找到相似度尽可能高的源域和目标域。

==(**TO DO**: 给了几篇比较general论文)==



**[CH PPT ](http://jd92.wang/assets/files/l08_tl_zh.pdf)中科院迁移学习简介**

*Why transfer learning?*:  标签很难获取；从头建模很麻烦。着重学习任务间**相关性**。

*简单例子*：with different instances, time period, data distributions, not need to have a lot of labels, reuse the model... :  **labeled training** on Source Domain data --> Transfer Learning Algorithms (FROM Target Domain Data with **Unlabeled** /a few labeled data for adaptation) --> **Predictive** Models --> **Testing** on Target Domain Data.  

给定：Domain $D_s$,$D_T$ ; Tasks $T_s, T_T$ ；目标：利用$D_s, T_s$学习$D_T$上的预测函数$f(·)$。限制：$D_s \ne D_T$ or  $T_s \ne  T_T$  

Notation: $\mathcal{D}=(X, P(X))$ (feature space, marginal distribution); Task: $\mathcal{T} =(Y, f(·))$, Y:label space, f(·): objective predictive function.  Goal: $min_\epsilon(f_T(X_T), Y_t)$ ; Conditions: Domain or Task different, Dt, Ds, Yt, Ys may be unknown. 

分类：

- 按情境 data distribution：

  - 归纳式 inductive 域的学习任务不同但相关；

    Ts不等于Tt，源域很多标签，或源域无标签

  -  直推式 transductive，域不同但相关，学习任务相同；

    T相同，Xs 不等于 Xt，或 P(Xs) 不等于 P(Xt)

  - 无监督 unsupervised，源域目标域都没有标签，域和任务不同但相关。

- 按特征空间：

  - 同构 homogeneous：特征维度相同，分布不同；
  - 异构 heterogeneous：特征维度不同。

- 按迁移方法 methodology，基于——：

  - 实例 instance based：通过权重重用双域的样例

    源域的一些数据和目标域共享了许多共同特征：对源域进行instance reweighting，筛选相似度搞的数据，进行训练。简单但度量依赖经验，双域数据分布往往不同！

    Reuse source domain: instance reweighting and importance sampling;

  - 特征 feature based：将双域特征变换到相同空间

    双域仅有一些交叉特征：特征变换，数据变换到同一特征空间，进行传统ML；大多采用，效果好；优化问题，难求解，overfitting。

    learn good feature representation of target domain;

  - 关系 relation based：利用源域的逻辑网络关系（如果两域相似，则会共享某种相似关系）

    relationships are same in source and target domains.

  - 模型 parameter based：利用双域参数共享模型 （这是基于模型层面，其他是基于数据层面。）

    共享一些模型参数，源域学到的模型用到目标域，再根据目标域学习调整。TRCNN。模型间相似，reuse；参数难收敛。

    transfer models between source and target domains.

研究领域：

- domain adaptation, cross-domain learning。有标源和无标Dt共享相同特征和类别，但特征分布不同——如何用Ds标定Dt 

  TCA，GFK，TKL，TransEMDT，Kernel mean matching，Covariate Shift Adaptation；Adaptive SVM or ASVM；Multiple Convex Combination or MCC

  假设双域数据（或在高维空间）有相同的条件分布，但相似性无法衡量（可能负迁移！）

- multi-source TL 多个源域和目标域，如果域筛选

  TrAdaBoost 过滤不相似样本再实例迁移；MsTL-MvAdaboost样本相似度+多视图学习的目标； Consensus regularization 在源域和未标注目标域训练分类器，利用一致性约束知识迁移； Transitive TL，相似度低的双域，利用第三方学习的相似度关系。Distant domain TL，相似度极低，autoencoder，多个中间辅助域中选择知识。

  利用多个可用域，效果较好；如何衡量多域相关性，又如何利用。

- 深度迁移：深度NN，NN学习NL的特征表示，层次性，uninterpretable data，表明在数据中有某些不可变的成分，可以用来迁移。

  > Deep Learning (NL Representations): hierarchical network and disentangle different explanatory factors of variation behind data samples. With TL (Alleviation减轻): don't need a lot of data.

  **Joint CNN**：针对稀疏标记的目标域，CNN同时优化域之间的距离和迁移学习task的损失loss (domain confusion loss, domain classifier loss, classification loss, softlabel loss) ；#??

  **SHL-MDNN**：不同学习网络之间共享隐藏层，用不同的softmax层实现不同人物。'shared feature transformation'

  **Deep Adaptation Network** (DAN): CNN中与学习任务相关的隐藏层 映射到 [再生核希尔伯特空间](https://www.google.com/search?q=%E5%86%8D%E7%94%9F%E6%A0%B8%E5%B8%8C%E5%B0%94%E4%BC%AF%E7%89%B9%E7%A9%BA%E9%97%B4)(~Kernel mean matching?)，进行多核优化，最小化不同域间的距离 #?? 

  **Joint Adaptation Networks**: 提出一种新的联合分布距离度量关系，利用其泛化模型的TL能力，以适配不同领域的数据分布。基于AlexNet和GoogleLeNet #??

  TL+DL：NICE！ TL大增强了模型繁华能力，DL可以深度表征 域的知识结构；

问题：负迁移（无法判断相关性！），没有统一的TL学习理论；没有统一有效的相似度衡量法。

**Negative Transfer**: Domains are too dissimilar; [Conditional Kolmogorov complexity]() is not related; Task not well-related (Transitive TL try to solve it by bridging the Ds and Dt using auxiliary sources.)

**Similarity Measure**: 1. Maximum Mean Discrepancy in Reproducing Hilbert Kernel space using kernel function like gaussian； 2. cosine similarity: $sim(X, Y) = \frac{X·Y}{|X||Y|}$; 3. Kullback-Leibler (KL) divergence; 4. Jensen-Shannon divergence (JSD);



针对问题已有的基础：Autoencode（相似度低域）；物理学定律作理论支撑；迁移度量学习，寻找行为间相关性最高的域进行迁移。

==(**TO DO**) 对自己可能有用的算法的论文。==

**[EN PPT, Introduction to Transfer Learning](http://jd92.wang/assets/files/l03_transferlearning.pdf)**

Definition: research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.

Conditions: Source and Target domains don't need to be in the same distributions.  Less training samples, even none. 

Notatin: $\mathcal{D}=(X, P(X))$ (feature space, marginal distribution); Task: $\mathcal{T} =(Y, f(·))$, Y:label space, f(·): objective predictive function.  Goal: $min_\epsilon(f_T(X_T), Y_t)$ ; Conditions: Domain or Task different, Dt, Ds, Yt, Ys may be unknown. 



[zhihu](https://www.zhihu.com/question/41979241)

## 王晋东TF简明手册

[王晋东TF简明手册](http://jd92.wang/assets/files/transfer_learning_tutorial_wjd.pdf)

### why TL? 

1. big data versus. few labels. --> 迁移数据标注（找与目标数据相近的有标注的数据）
2. big data versus. weak computation --> 模型迁移， 训练好的大模型+微调
3. 模型泛化能力 vs 个性化需求。 自适应学习，对普适化模型进行灵活调整
4. 特定应用的需求：如崭新的系统，没有足够的标签。 相似领域知识迁移/数据迁移。

**区别和联系**

1. TL - 传统ML
2. TL - 终身学习 life-long 连续不断单域学习
3. TL - 多任务学习 multi-task 多个任务同时完成
4. TL - 领域自适应 domain adaptation, 子类
5. TL - 增量学习 incremental 一个域上不断学习
6. TL - 自我学习 self-taught 从自身数据学习
7.  TL - 协方差漂移 covariance shift，子类
8.  ‘负迁移’ --解决：传递TL，TransitiveTL：使用相似度低的双域间的若干第三方域。

**分类** 见上面笔记。

1. 按目标域(有无)标签分：supervised / semi-supervised / unsupervised
2. 学习方法： instance / feature / model / relation
3. 特征：homogeneous / heterogeneous 如果特征语义和维度相同，同构（不同图片的迁移）；图片到文本——异构。
4. 离线 offline TL：双域都给定，迁移一次即可，但无法对新加入的数据进行学习，模型也无法得到更新 / 在线 online TL：数据动态加入，TL算法也可以不断更新。

**应用** 

1. 计算机视觉：TL被称为Domain Adaptation——图片分类，哈希。如同类图片，不同角度，背景，光照造成特称分布改变。TL构建跨领域的鲁棒分类器。见三大CV顶会：CVPR，ICCV，ECCV。
2. 文本分类
3. 时间序列（行为识别 activity recognition 不同用户、环境、位置、设备导致的时间序列数据分布变化。）（Indoor Location，Wifi，蓝牙，不同用户、环境、时刻、设备）
4. 医疗健康 etc。针对无法获取足够有效的医疗数据这一问题。



### 基础知识

**形式化** 

- Domain领域：进行学习的主体（数据:**x**向量(x_i=i个样本/特征)或**X**矩阵或$\mathcal X$表示数据的特征空间，生成这些数据的概率分布:*P*）；源域：有知识，大量标注；目标域：赋予知识、标注的对象。P为一个逻辑上的概念，很难给出具体形式。
- Task任务：（标签$\mathcal Y$, 标签对应的函数/学习函数 f(·)）
- 迁移学习：给定一个有标记的源域$\mathcal D_s =(x_i, y_i)_{i=1}^{n}$ 和一个无标记的目标域$\mathcal D_t =(x_j)_{j=n+1}^{n+m}$,样本数m。两域数据分布$ \mathit P(x_i) \ne \mathit P(x_j)$（或称边缘分布） .借助D_s的只是，来学习D_t的标签（知识）。需考量
  - 特征空间X的异同
  - 类别空间Y的异同 
  - 条件概率分布的异同：$Q_s(y_s|x_s) \ne Q_t(y_t|x_t)$ ?
- Domain Adaptation: 给定有标源域和无标目标域，假定他们的特征空间X相同，类别空间Y也相同，但两域边缘分布不同，即P不同，条件概率分布也不同，即Q不同。利用D_s去学习一个分类器f:x_t --> y_t（要学习的目标函数）来预测D_t的标签y_t（标签属于类别空间$\mathcal Y$） #??
- 相似性（不变量）是核心，度量（相似性的）准则是重要手段（定性定量，以此为准，采用学习手段增大两域相似性）

**度量准则**

描述源域和目标域的距离 $DISTANCE(\mathcal D_s, \mathcal D_t) = DistanceMeasure(·,·)$

*常见距离*：

1. 欧式距离 d_Euclidean p=2
2. Minkowski distance: p阶距离 p=1 -- Manhattan
3. Mahalanobis 定义在同一分布里的两个向量上：协方差的逆矩阵，协方差为I时，为欧氏距离。

*相似度*：

1. 余弦相似度 cos(x,y) = xy / |x||y|
2. 互信息 mutual information
3. Pearson coefficient 协方差矩阵除以标准差之积 [-1,1] 绝对值越大(正/负)相关性越大。
4. Jaccard coefficient：集合的交集除以并集， jaccard距离 = 1- J

广泛使用：

* KL散度*： Kullback-Leibler divergence:  相对熵，衡量P和Q的距离，非对称距离。*
* JS距离*：Jensen-Shannon divergence, 基于KL，对称度量。P，Q，引入M=1/2(P+Q).

*最大均值差异 MMD*（使用频率最高）：Maximum mean discrepancy，度量了两个分布在再生希尔伯特空间的距离，是核学习方法。其中$\sigma(.)$是映射，把原变量映射到RKHS中。即是求两堆数据在RKHS中的*均值*的距离。

又有Multiple-Kernel MMD, or MK-MMD。假设最优的核由多个核线性组合得到。在许多方法中大量使用 —— 最著名：DAN。

**Principal Angle**: 将两个分布映射到高维空间中，两堆数据就可看成两个点。求这两堆数据的对应维度的夹角之和。对于两个矩阵X，Y，先正交化(PCA)，然后求夹角和，夹角为矩阵SVD后的角度。 #??

**$\mathcal A$-distance** 可用来估计不同分布的差异性。被定义为：建立一个线性分类器来区分两个数据域的hinge loss(binary). 首先在双域训练一个二分类器h，使得h可以区分样本来自那个域，在用err(h)来定义h的loss。A-distance：A(D_s, D_t) = 2(1-2err(h)). 用来计算两域数据相似性程度，以对结果验证对比。

**Hilbert-Schmidt Independence Criterion or HSIC** 检验两组数据的独立性：X，Y为两堆数据的kernel形式。 HSIC(X, Y) = trace(HXHY)



### 基本方法

**基于样本**：instance reweighting，如认为提高某重要相似类别的样本权重。P很难估计，着眼于估计双域分布的比值Pt/Ps. 此比值为‘样本的权重’。经典研究：TrAdaBoost方法；Kernel Mean Matching KMM方法，使加权后源域和目标域的概率分布尽可能相近；Transitive TL TTL方法， DDTL方法等。

良好的理论支撑，容易推泛化误差上界。一般只在领域分布差异确实较小的时候有效，对自然语言，cv不理想。

**基于特征**：效果好，是重点，热门。将通过特征变换的方法互相迁移，来减少双域间的差距；或将双域的数据特征变换到统一特征空间，再用传统ML进行分类识别。可分为同构和异构。通常假设双域间有些交叉的特征。如TCA方法，用MMD度量，将不同域的分布差异最小化。SCL方法：映射一个空间中独有的一些特征变换到其他所有空间中的轴特征上，然后在该特征上ML分类。TJM，实例+特征迁移结合。多与神经网络结合。

**基于模型**：找到双域间共享的参数信息。假设双域的数据可以共享一些模型参数。多与深度神经网络结合，有些方法对现有的一些网络结构进行修改，加入domain adaptation层 #??，联合进行训练等，可以看作模型+特征方法结合。

**基于关系**：关注双域样本之间的关系如生物病毒--计算机病毒，如借助markov logic net。

#### 第一类方法：数据分布自适应

##### A. 边缘分布自适应 Marginal Distribution Adaptation

Distance $\approx ||P(x_s) - P(x_t)|| $ 

经典方法：**TCA, Transfer Component Analysis**, 找到一个映射$\phi$, 使得$P(\phi(x_s))\approx P(\phi(x_s))$, 那么条件分布也会接近$P(y_s|\phi(x_s))\approx P(y_t|\phi(x_t))$. 使用MMD为距离度量，用Kernel矩阵求解...

##### B. 条件分布自适应 

拉近条件分布的差异。**STL**方法，在绝大部分方法只学习全局特征变换（Global Domain Shift)的基础上加上了Intra-class Transfer, 利用类内特征实现更好的迁移。

##### C. 联合分布自适应 Joint Distribution Adaptation

JDA：找到一个变换A，使变换后边缘分布和条件分布都更接近。

BDA：平衡因子$\mu$, 调整两种分布距离的重要性。

BDA > JDA > TCA > 条件分布自适应。

#### 第二类: 特征选择

假设双域有一些公共的特征，在这些特征上双域的数据分布一致。找出这些特征，根据其构建模型。

**SCL** 找到公共特征“pivot feature”；

常与分布自适应方法结合

#### 第三类：子空间学习

双域数据在变换后的子空间中有相似的分布。

**统计特征对齐**。 

**SA**（Subspace Alignment）。找一个线性变换M，使不同数据实现变换对齐。$F(M)=||X_sM-X_t||^2$ 

**SDA**: SA+概率分布自适应。子空间变换矩阵T+概率分布自适应变换A——$M=X_sTAX_t^T$ 

**Coral**：Corelation alignment，对于两域的协方差矩阵C_s, C_t: 寻找二阶特征变换A，使两域特征距离最小。神经网络：DeepCORAL，将CORAL度量作为一个神经网络的损失计算。$min||A^{T}C_sA-C_t||^2$ 

**流形学习**。

假设数据是从一个高维空间中采样出来，具有高维空间的低维流形结构（在高维空间里有形状）。*地球上两点之间，测地线（在球体上是一条曲线）最短。* --> GFK方法。

首先，将原始空间的特征 变换到 流形空间中。 如Grassmann流形$\mathbb{G}(d)$将原始数据的d维子空间（特征向量）看作它基础的元素，帮助学习分类器；其中，特征变换和分布适配都有着有效的数值型式。

**SGF方法**：源域和目标域为高维空间G流形的两个点，从一点走到另个点：在两地测点线距离取d个中间点，依次连接形成路径。找到合适每一步的变换，即可从源域变换到目标域。*需要几个中间点？*

**GFK方法**：多少个点？——核学习，将路径上无穷个点积分。多个源域时怎么办？——提出Rank of Domain度量，使用跟目标域最近的源域。



--------------------------------

### 深度迁移学习

直接对原始数据进行学习：优势：*自动化提取*更具表现力的特征，满足实际应用中端到端*end-to-end*的需求。

*现在对抗网络GAN对比传统深度神经网络极大提升了学习效果。*

***为什么深度网络可以迁移？***

层次结构：前面几层学到的是general feature，后面更偏重与task特定的specific feature。那么应该迁移哪些层，固定哪些层？——分析论文[Yosinski et al., 2014] Yosinski, J., Clune, J., Bengio, Y., and Lipson, H. (2014). How
transferable are features in deep neural networks? 

神经网络前三层是general feature，进行迁移效果好；深度迁移+finetune，效果提升大，可能比原来还好。Fine-tune可以比较好克服数据之间的差异性。深度迁移比随机初始化权重好。网络层数迁移可以加速学习和优化。

***最简单的深度迁移：Finetune***

利用别人已经训练好的网络，（但不是完全适用于自己的任务，比如数据分布不同，别人网络可以做更多事，网络更复杂等）针对自己的任务进行调整。如猫狗的二类分类可以直接改CIFAR100训练好的网络（100个类别，我们只需要两个，固定相关层，修改输出层。）

Fine-tune的拓展：如替代传统的人工提取特征方法。使用深度网络训练raw data，提取更有表现力的特征，再用这些特征输入传统的ML方法。 CV领域：DeCAF特征提取方法：用ConvNet进行特征提取，精度好很多。也有用CNN提取的特征作为SVM输入，提升分类精度的方法。

#### **深度网络自适应**

finetune无法处理<u>训练数据和测试数据分布不同</u>的情况。它的基本假设是train和test数据分布相同。这个在TL中不成立的。

参考数据分布自适应方法，许多深度学习方法都开发出了**自适应层Adaptation Layer**来完成源域和目标域数据的自适应，使两堆数据分布更加接近，达到更好的网络效果。深度网络的自适应的任务——

1. 哪些层可以自适应：决定了网络学习程度
2. 采用什么样的自适应方法（度量准则）：决定了网络的泛化能力。

深度网络的最终损失 = 在有标注的数据上（多为源域）的常规分类损失 + $\lambda$*网络的自适应损失（此为TL独有). $\lambda$为权衡两部分的权重参数。$\mathcal l = \mathcal{l_c}(D_s, y_s) + \lambda \mathcal{l_A}(D_s, D_t)$ 

基本准则：决定自适应层，在这些层加入自适应度量，最后finetune。

**核心方法**

DaNN(Domain Adaptive Neutral Network, 2014)。两层：特征+分类器+MMD适配层（测量两域距离，并加入网络损失训练）。太浅，表征能力有限，\domain adaptation

**DDC**(Deep Do)main Confusion, 2014)：基于AlexNet，前七层固定，第八层（分类器前一层）加自适应的度量（MMD）。损失函数为分类损失和$\lambda$MMD^2^损失。why 8th? 通常分类器前一层为特征，在特征上加上自适应正是迁移学习的工作。

$l = l_c(\mathcal D_s, y_s) + \lambda MMD^{2} (D_s, D_t)$

**DAN**(Deep Adaptation Networks, 2015)：扩展DDC，基于AlexNet，分类器前三层加*三个自适应层*，使用表征能力更好的*MK-MMD*（参数beta）。

优化目标也由两部分：损失函数cross_entropy+自适应损失(基于MK-MMD距离)。

$min\frac{1}{n_a} \sum_{i=1}^{n_a}J(\theta({x_i}^a), {y_i}^a)+\lambda \sum_{i=l_1}^{l_2} {d_k}^2(D_s, D_t) $  --> weight, bias.

学习两类参数：网络参数（w和b）和beta。<u>学习w，b</u>：依赖MK-MMD距离，O(n^2^). 采用Gretto n的MK-MMD无偏估计，降为O(n). 进行SGD，要对所有w和b求导，mutiple-kernel运用的是多个高斯核。学习beta：确定多个kernel的权重，确保每个kernel生成的MMD距离方差最小。

Joint CNN for domain and task transfer，DDC拓展，同时迁移domain和task：

- domain transfer 适配分布：特别指适配marginal distribution，但没有考虑类比信息。how：传统loss + confusion loss（classifier能否将两个domain分开的loss）--> domain transfer
- task transfer 利用class之间的相似度。特指conditional distribution。即作者认为只考虑domain classifier+domain confusion不够，还要考虑类别间的联系（杯子和瓶子比和键盘更相似），所以要再加一个soft label loss。即在源域和目标域上进行适配时，要根据源域的类的分布来调整target的类的分布。~JDA。基于AlexNet：loss 由普通训练loss + domain Adaptation loss+ soft label loss(only act on target)构成。
- $ L = L_C(x_s, y_s, x_T, y_T; \theta_repr, \theta_C) + \lambda L_{conf}(x_s, x_T, \theta_D; \theta_repr) + \nu L_{soft}(x_T, y_T; \theta_repr, \theta_C)$ 
- 什么是soft label loss: 目标域没有多少label，利用源域的label，在对源域训练时把每个样本处于每个类的概率记下，算出所有样本属于每一个类的概率（求和平均），利用这个分布关系对target做相应的约束。

**JAN** Joint Adaptation Network 2017 深度联合分布自适应：将只对数据adapt的方式推广到了对类别的自适应。JMMD度量。

**AdaBN**：Adaptive Batch Normalization 2018，在归一化层加入统计特征的适配。没有额外的参数。Input——Conv，FC——BatchNorm——Activation——Output。

**小结**：核心为——找到网络中需要自适应的层，并加上自适应的损失度量。多用CNN的AlexNet，Inception，GoogleLeNet，Resnet等迁移。自动深度网络自适应层：AutoDIAL实现了自动的自适应学习。

#### **深度对抗网络迁移： on GAN, Generative Adversarial Nets**

**GAN**: 二人零和博弈——两部分：生成网络，负责生成以假乱真的样本，*Generator*；判别网络，判别样本是真实的还是生成的，*Discriminator* $G_d$。二者互相博弈，完成对抗训练。

在迁移学习中，可以将一个域的数据（多为*目标域）当作是生成的样本*。生成器此时不生成，而是做<u>特征提取</u>：不断学习该域数据的特征，使得判别器无法对双域进行分辨。这样生成器变成了$G$ --> Feature Extractor $G_f$。损失为：网络训练损失和领域判别损失。 $l = l_c (D_s, y_s) + \lambda l_d(D_s, D_t)$ 

   	1. **DANN** Domain-Adversarial Neural Network 2016. 生成的特征尽可能帮助区分两域特征，同时使得判别器无法对两域的差异进行判别。
   	2. **DSN** Domain Separation Networks 2016. 两域都由公共和私有部分组成。公共：学习公共特征；私有：保持各域独立的特性； $l=l_{task}+\alpha l_{recon} + \beta l_{difference} + \gamma l_{similarity}$
   	3. **SAN**针对源域存在很多目标域没有的类别，导致负迁移——partial transfer learning，针对目标域选择相似的源域样本/类别。

## A Survey on Transfer Learning, Pan, Yang

1. **Intro**
2. **Overview**
   1. History
   2. Notations and Definitions
   3. Categorization
3. **Inductive Transfer Learning**
   1. transfer Knowledge of Instances
   2. Knowledge of Feature Representations
   3. Knowledge of Parameters
   4. Relational Knowledge
4. **Transductive Transfer Learning**
   1. Instances
   2. Feature Representations
5. **Unsupervised Transfer Learning**
   1. Feature Representations
6. **Transfer Bounds, Negative Transfer**
7. **Applications** 
8. **Conclusions**

## How transferable are features in deep neural networks?

How to quantify | Transition occur suddenly/spread out? | Where does it occur?

(**Section 2**): Tendency of Gabor filters and color blobs in the first layer. experimentally quantify the generality vs. specificity of neurons in each layer of a deep convNet. iterate n in AnB, BnB, AnB+(finetuned), BnB+.

Whether to fine-tune the first n layers? depending on size of target dataset and the number of parameters.. .是否finetune迁移层的参数——

small size, many parameters: may overfit. often left frozen. 参数太多，小数据集不finetune，防止overfit。

large size, few parameters: base features can be fine-tuned to the new task. (if very large, TL may not be necessary.)参数不多，大数据集

(**Section 4)**: train pairs of ConvNet on ImageNet and characterize the layer-by-layer transition from general to specific, which yields the following 4 results.

(**Section 4.1**): Transferability affected by: (1) the specialization of higher layer neurons to their original task at the expense of performance on the target task. (2) optimization difficulties related to splitting networks between co-adapted neurons.  Either may dominate (at different layers of network, aka. depending on transferring from bottom, middle or top. ) 中间层层间联系紧密，高层特异化较强，都会影响 迁移。

(**Section 4.2** dissimilar datasets: Man-made and natural classes in separate datesets): Transferability of features decreases as the distance between base task and target task increases (more dissimilar the base task and target task are). 越不相似(人造-自然)越难迁移。

(**Section 4.3** Random weights): but still better (frozen and fine-tuned) than random lower-layer weights.不管如何迁移都比随机生成权重好。

(**Section 4.1** similar datesets: random A/B splits): Initializing with transferred features from almost any number of layers can produce a boost to generalization that lingers even after fine-tuning to the target dataset.（相似数据组：随机分开的数据，迁移会带来generalization的上升）

(**TO DO** References)

## Deep Learning of Representations for Unsupervised and Tbransfer Learning,  Yoshua Bengio

Deep Learning seek to exploit the unknown structure in the input distribution to get good representations. Objective: make higher-level representations more abstract*(--> more invariant to most of the variations in the training distribution, preserving as much as possible of the information in the input)*, to disentangle the unknown factors of variation in the training distribution. Unsupervised Learning of Representations Under Hypothesis: input distribution P(x) related to some task of interest, say predicting P(y|x). 

Focus: why unsupervised pre-training of representations can be useful, how to exploited it in TL, where we care about predictions on examples that are not from the same distribution as the training distribution. 

 #### Intro

the way in which data are represented can make a huge difference in the success of a learning algorithm. Why good representation, What is a good representation, How to discover good representation. 

'Unsupervised':*labels for the task are not available at the time of representation learning.* Wishes: learn it in purely unsupervised way, or using *labels for other tasks*. Related to: self-taught learning, TL, domain adaptation, multi-task learning, semi-supervised learning... 

'Deep Learning': learning multiple levels of representation, discovering more abstract features in the higher levels of representation, which may make it easier to separate from each other the various explanatory factors extent in the data.

**Unsupervised + TL Challenge**

main difficulties:

- input distribution very different in test set compared to training set(e.g classes in test set may be absent or rare in the training set). unclear: if anything can be transferred?
- very few labels available to the linear classifier on the test set, make generalization of the classifier difficult.
- No labels for the classes of interest(of the test set) are available when learning the representation.  labels from the training set might mislead the representation learning.

great pressure on the representation learning algorithm on training set(unlabeled) to discover really generic features.

## Understanding Neural Networks through Deep Visualization, Yosinki

(1) provides a tool that visualizes the activations produced on each layer of a trained convnet. 

1. representations on some layers seem to be surprisingly local?
2. webcam image: imprecise, the probability vector is noisy, varies much via tiny changes.  (sensitivity to small changes)
3. the last 3 layers are sensitive, lower layer computation is more robust.

(2)visualizing features at each layer of a DNN via regularized optimization in image space.  introduces some regularization methods that produce more interpretable visualization.

Approach to understand what computations are performed at each layer:

- study each layer as a group
- interpret the function computed by each neuron:
  - dataset-centric: display images that cause high activations for individual units/deconvolution method.
  - network-centric: without any data. synthesize inputs that cause higher and higher activation of unit i using gradient ascent. But the generated images are composed of "hacks" that happened to cause high activation, which can cause correlty classified images to be misclassified even via small changes. Hence using regularization to produces useful images. 
  - Regularization used: L2-decay, Gaussian blur, Clipping pixels with small norm, clipping pixels with small contribution; combined effect produces better visualizations, (random hyperparameter search)

Conclusion:

- the locality of representations on later conv layers (where channels correspond to specific, natural parts) suggests that during transfer learning, when new model trained atop con4/5, a bias toward sparse connectivity could be helpful.
- discriminative parameters also contain significant 'generative' structure from the training dataset. even with simple p(x) model as regularizer,... (might be helpful: transferring discriminatively trained parameter to generative models! 有点反直觉)



## Learning and Transferring Mid-Level Image Representations using ConvNet

reuse layers trained on ImageNet to compute mid-level image representation for images in PASCAL VOC dataset. despite different image statistics and tasks, the transferred model leads to better results for object and action classification.



## Low Shot Visual Recognition by Shrinking and Hallucinating Feature

(1) Representation regularization techniques;

(2) Techniques to hallucinate additional training examples for data-starved classes. 

related to: One-Shot/low-shot learning, Zero-shot learning, Transfer learning.

**A low-shot learning benchmark** employs:

<u>a Learner</u>: feature extractor + a multiclass classifier

<u>two training phases</u>: 

1. Representation learning: D + Cbase --> feature extractor. (ConvNet, cross-entropy loss)

2.  low-shot learning: C^l^ = C_base + C_novel, D, feature extractor from 1. --> multi-class classifier, (optionally modifying feature extractor) (train a new linear classifier head)

<u>a testing phase</u>: predicts lables from C^l^.

<u>two improving strategy</u>: 

1. hallucinating additional training examples

   *train a function G*: take concatenated feature vectors of x, z_1, z_2 as input; produces as output a "hallucinated" feature vector. (MLP, 3FC)

   *generated examples*: use them when the number of real example in novel class is really low(lower than a hyperparameter k).

2. improving representation itself.

   *Goal*: reduce the difference between classifiers trained on large D and on small D, so that those trained on small datasets generalize better. 

   1. using Squared gradient magnitude loss or SGM. (aka using a new loss function) (singleton sets for S, or batch SGM)
   2. or Feature regularization-based alternatives. (using simple squared L2 norm as loss, aka regularizing the *feature representation, not the weight vector!* )
   3. or Metric-learning based approaches. (triplet loss).

<u>Result</u>: 

Representation:

1. accuracy on base classes does not generalize to novel classes, especially when novel classes have very few training examples.
2. with regularization always better. L2 better for mall n. 
3. ...
4. finetuning the representation not only expensive, but does not help.

Generation: produces more diverse and useful training examples than simple data augmentation.

our contributions generalize to deeper, better models.