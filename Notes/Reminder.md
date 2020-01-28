[toc]

## 分词

* 最大匹配
* n-gram模型（语言模型）
* 维特比算法



## 拼写纠错

* 编辑距离（edit distance），有三种操作，假设他们都是一个单元的距离成本
  * insert
  * delete
  * replace
* 使用编辑距离计算语句的相似度



## DP算法(动态规划)

* Big problem ---- small problem
  * 分治：将原问题划分为互补相交的子问题，递归的求解子问题，再将他们的解组合起来
  * 最优子结构：问题的最优解由相关子问题的最优解组合而成
  * 边界：问题的边界，得到有限的结果
  * 动态转移方程：问题每一阶段和下一阶段的关系
    * 1. 问题中的状态满足最优性原理 —— 最优子结构
      2. 问题中的状态必须满足无后效性 —— 以前出现状态和以前状态的变化过程不会影响将来的变化
    * Example：
      1. Fibonacci（合并状态）
      2. 有n个石子，AB轮流取石子，最少一个，最多两个，取最后一个的人赢

### 背包问题

$N$件物品，容量为$W$的背包。第i件物品的种类为$w_i$，价值是$V_i$。求装的最大价值。
$$
dp[i, j] = max \{ dp[i-1, j], dp[i-1, j-W_i] + V_i \}
$$
以上是取到前$i$件物品，容量为$j$的最大价值。![image72fa11c630bcfd5a.png](https://file.moetu.org/images/2020/01/16/image72fa11c630bcfd5a.png)

###  完全背包

![imagead328fe29cd4e305.png](https://file.moetu.org/images/2020/01/16/imagead328fe29cd4e305.png)



## 相似度比较

* 通过用户输入生成编辑距离为1，2的单词
* 过滤
* 返回

![image-20200116203545895](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200116203545895.png)



## 停用词过滤

* nltk中停用词库的使用
* 低频无关词汇的去除



## 标准化

* stemming：one way to normalize
  * 不保证转换成有效的原型
  * [使用最广泛的stemming工具](http://tartarus.org/martin/PorterStemmer/java.txt)
* lemmazation
  * 比stemming更加的严格，可以保证转换出的原型在字典中符合语法

![imagef2a3d34c003d09c8.png](https://file.moetu.org/images/2020/01/16/imagef2a3d34c003d09c8.png)

## 文本表示

* one-hot编码
* 向量的维度等于词典的大小
* boolean-base representation
* count-base representation



## 句子相似度的计算

* 欧式距离 
* ![image06b5b99b29f6f567.png](https://file.moetu.org/images/2020/01/17/image06b5b99b29f6f567.png)
* 余弦相似度（使用的更多更广泛）
* ![imagecadae7be4e0686e5.png](https://file.moetu.org/images/2020/01/17/imagecadae7be4e0686e5.png)



## Tf-idf文本表示

* 反映单词的重要程度
* ![image49f90c0af1046c78.png](https://file.moetu.org/images/2020/01/17/image49f90c0af1046c78.png)
* 1. 构造一个字典
  2. 将句子转换为tf-idf向量（向量长度也等于字典的长度） 



## one-hot编码存在的问题

1. 不能表示语义的相似度
2. 一般情况下都是稀疏矩阵，后续处理存在困难



## Distributed Representation（分布式的表示方法）

* 向量的长度是自定义的，一般要小于one-hot的表示长度
  * 解决了稀疏矩阵的问题
* 每一个位置都是一个非零的数值（小于1）
  * 同样使用欧式距离、预先相似度等方式来计算
* 分布式表示方法针对于单词的叫做词向量（word vector）
* n维的one-hot最多可以表示n个单词；n维的分布式表示法最多可以表达至少$2^n$个（准确说是无穷个）
* 简单流程：
  * 输入（input）：string
  * 深度学习模型：Skip-Gram、Glove、CBow、RNN/LSTM、MF
  * dim/D：设置好维度



## 词向量

* 词向量代表单词的意思（meaning）
* word2vec：某种意义上可以理解成词的意思



## 如何表示一个句子的向量

* 平均法（Average）

  * 将句子中每个词词向量相加并求平均

* LSTM/RNN（深度学习）

* 将句子的向量计算出来以后再通过余弦相似度或欧式距离等方式来判断两个词的相似度

  

## Retrieval-based QA System

* 问答库的数据量可能非常的大，在这里就要用到“层次过滤思想”
* 层次过滤的方法复杂度需要由简单到复杂



## Inverted Index（倒排表）

* 每个单词后对应了这个单词出现的文章
* 搜索引擎一般会使用到倒排表
* 可以利用倒排表来优化简单的问答系统
  * 通过倒排表来优化问答系统的过滤搜索过程

## Noisy Channel Model（噪音通道模型）

![image1a91ed9dac67c7e5.png](https://file.moetu.org/images/2020/01/18/image1a91ed9dac67c7e5.png)

![imagec250cf8d48a3a5c5.png](https://file.moetu.org/images/2020/01/18/imagec250cf8d48a3a5c5.png)



## 语言模型

* 可以用于求解一个句子出现的概率

  * 涉及到了概率中的链式法则（Chain Rule）

* Unigram：假定每个词都是一个独立的个体

* Bigrame：来源于1rd马尔科夫假设，假设每个词语他的前一个词有关

* Trigrame：来源于2rd马尔科夫假设

* 评估：Perplexity（无监督）

  ![image4d58ee345b63b825.png](https://file.moetu.org/images/2020/01/18/image4d58ee345b63b825.png)

* 在N-gram模型中，n越大perplexity越小，但是一定程度上也越容易出现过拟合。



## Markov Assumption

![image83e4af3c6b2c03d0.png](https://file.moetu.org/images/2020/01/18/image83e4af3c6b2c03d0.png)

* 1rd：每个单词与前一个单词有关
* 2rd：每个单词与前两个单词有关
* 3rd：每个单词与前三个单词有关
* 语言模型分别对应Unigram，Bigram，Trgram



## 平滑系数（smoothing）

* 在出现了语料库中没有出现的单词时，如果直接使用乘法就会出现0，为了避免这一种线性我们会考虑平滑的方法

* 1. Add-One Smoothing

     * 拉普拉斯平滑：在贝叶斯计算的过程中上面加1下面加V（总次数）

       ![imagef9984a151411dc8c.png](https://file.moetu.org/images/2020/01/18/imagef9984a151411dc8c.png)

  2. Add-K Smoothing

     * ![image5396b832b74f1d9d.png](https://file.moetu.org/images/2020/01/18/image5396b832b74f1d9d.png)

  3. Interpolation

     * 把多种N-gram的概率综合起来考虑，给每个方法一个权重，注意这些权重的和为1
     * 实质是一种加权平均
     * 主要是因为现在不出现不代表未来也不会出现

  4. Good-Turning Smoothing

     ![image500c827c03b55a6e.png](https://file.moetu.org/images/2020/01/18/image500c827c03b55a6e.png)

     * 有一个致命的缺点：就是在高频次的区间中，有些词可能不再出现，在这里就需要先对这些高频段做缺失值处理才能求出原来我们所要的期望值



## Main Branch of Learning

* 专家系统
  * 基于规则（符号主义）
  * 没有数据或很少量
* 基于概率的系统
  * 基于概率统计（连接主义）
  * 大量数据



## 专家系统（Expert System）

* 专家系统 = 推理引擎（可以包含智能的部分） + 知识 （类似于程序 = 数据结构 + 算法）
* 利用**知识和推理**来解决决策问题（Decision Making Problem）
* 全球第一个专家系统叫做DENDRAL，由斯坦福大学学者开发于70年代
* Working Flow
  1. Domain Expert
  2. 输出经验
  3. Knowledge Engineer
  4. 存储到知识库（Knowledge Base）
  5. 算法工程师设计推理引擎
* * 可以处理不确定性
  * 知识的表示
  * 可解释性
  * 可以做知识推理
* 缺点
  1. 设计大量的规则
  2. 需要领域专家来主导
  3. 可移植性差
  4. 学习能力差
  5. 人能考虑的范围是有限的
* 可能面临的问题
  * 逻辑推理
  * 规则的冲突
  * 选择最小规则的子集



## POS-Tagging（词性标注实战）

![imagecd04d61b0e407a2b.png](https://file.moetu.org/images/2020/01/19/imagecd04d61b0e407a2b.png)

* 根据噪音通道模型可以看出，想要判断出这个句子里的单词所属的最大概率的词性，可以将其转变为翻译模型（Translation Model）和语言模型（Language Model）的乘积，语言模型暂取Bigram模型
* 取对数操作将连乘转换成累和的形式，分别代写为A，B，π
* A：
  * M*N的矩阵，M是词汇总量（词库），N是词性总量
  * 元素表示在出现词性N时，每个词语出现的概率$A_i$
* π：
  * 是一个向量
  * 大小N是向量的大小，他表示了以某一词性X作为句子第一个单词的概率为多少
* B：
  * 状态转移矩阵
  * N * N的矩阵，N是词性的总量
  * 元素表示从词性$Z_1$到$Z_2$的转变的概率

* **程序思想:**
  
  1. 考虑到我们后面都是使用对应的下标来取字典中的词（对于词性也是使用相同的操作），所以我们在处理数据时要先建立他们与下标对应的字典
  
  2. 使用NumPy定义π，A，B：
  
     π使用`np.zeros(N)`初始化一个长度为N（number of tags）的元素为0的向量；
  
     同理，A定义`np.zeros((N, M))`矩阵A，N（number of tags），M（number of words in dictionary）；
  
     B定义`np.zeros((N, N))`矩阵B，N（number of tags），表示词性状态转换矩阵 ，表示由状态i转换为状态j的概率
  
  3.  计算π：
  
     定义一个`pre_tag = ''`表示这是一个句子的开始，如果这是一个句子的开始，就把`pi`这个N维向量中对应的`tag`的值加一，即`pi[tagId] += 1`
  
  4. 计算A：
  
     在做完上一步的时候，紧接着可以`A[tagId][wordId] += 1`，表示对应这个`tagId`的tag下对应的word的计数加一
     
  5. 如果不是句子的开头，就对应A加一，然后对于词性状态转移矩阵`B[tag2Id[prev_tag]][tagId]`从前一个状态转换到下一个状态的计数加一
  
  6. 如果是`item[0] == '.'`则表示句子到了结尾，将`prev_tag`重置为空，否则存入当前的词性`prev_tag = item[1].rstrip()` 
  
  7. 加下来我们通过简单的数学操作把这些次数转换为概率
  
  8. 接下来使用动态规划的**状态转移方程**，求出概率最大的路径（维特比算法！重点！！！）
  
  9. 定义维特比算法：（画图有助于理解！）
  
     * 对于输入的sentence进行切分，并返回对应单词的ID
  
     * 定义一个`dp = np.zeros((T, N))`，每列是词库，每行是对应的词性tag
  
     * 由于维特比算法递推需要知道第一行，故先定义好第一行的值：
  
       ```python
       for j in range(N):
       	dp[0][j] = log(pi[j] + log(A[j][x[0]])
       ```
  
     * 接下来就是对与剩下每一个单词对应的每一个词性的概率求值（定义一个数组存储路径）：
  
       ```python
       np.array([[0 for x in range(N)] for y in range(T)])
       for i in range(1, T):
       	for j in range(N):
       		dp[i][j] = -99999999
       		for k in range(N)
       		score = dp[i-1][k] + log(A[j][x[i] + log(B[k][j])
       		if score > dp[i][j]:
       			dp[i][j] = score
                    ptr[i][j] = k         
       ```



## 分类器的评估标准

1. 准确率（precision）

   * 预测正例中，真实为正例的数量
   * 当样本不平衡时不适合使用准确率，比如医院病患概率预测

2. 召回率（recall）

   * 真正样本中，预测为正例的比例

   ![image5939fa04340c84a7.png](https://file.moetu.org/images/2020/01/21/image5939fa04340c84a7.png)

   

## F1-Measure

![imageb5bcd4bb4a8d9385.png](https://file.moetu.org/images/2020/01/22/imageb5bcd4bb4a8d9385.png)

* 反映了模型的稳健性



## Logistic Regression（逻辑回归）

![image5661fb0f0205a384.png](https://file.moetu.org/images/2020/01/22/image5661fb0f0205a384.png)

![image4fb453b24df7b72a.png](https://file.moetu.org/images/2020/01/22/image4fb453b24df7b72a.png)

* 判断一个模型是不是线性的主要在于判断这个模型的决策边界（Decision Bound）是线性还是非线性的



## 模型的实例化（确定w和b）

* 模型的实例化相当于定义一个明确的目标函数
* ![image038ea20b4f108daa.png](https://file.moetu.org/images/2020/01/24/image038ea20b4f108daa.png)



## 梯度下降

* 在其实凸函数的前提下，可以使用优化算法
* 优化算法：（类似于循环一步步找到最优值）
  * GD
  * SGD（数据量巨大时）
  * Adagrand
* 1. 随机初始一个w值和b值
  2. 开始循环，每次循环中开始更新w和b
  3. 设置好学习率（步长一般会越来越小）
  4. 经过足够多的循环之后就可以到达那个近似点
* 什么时候停止
  1. 考虑用真实函数与之作比较，当误差小于定值时可以停止迭代
  2. 考虑两次迭代之间w和b的插值，小于定值时可以停止迭代
  3. 考虑通过一个验证集来验证他的准确性（early stopping）
  4. 考虑设定一个指定的次数让他循环（fixed iteration）



## 随机梯度下降

* 随机梯度下降
  * 首先有一个外层的循环	
  * 在内层循环前有一个`shuffle`操作，将样本重新排列
  * 内层循环样本次（n次）
  * 这样可以使得每一次循环都会更新w和b（但是步长会比之前的更小）
* Mini-batch Gradient Descent，在所有的n个样本里选择m个样本
  * 对每一个mini-batch进行操作
* 对比：
  * 梯度下降比较稳定
  * 随机梯度下降受噪声影响较大
  * Mini-Batch比较折中，但是每次选的batch的大小需要注意



## 逻辑回归中线性模型趋向无穷大问题

* 假设我们的数据是线性可分的，那就会导致w变成无穷大

  * 线性可分指的是类似二分类这种可以完美分割的

* 这时候就会考虑用到正则化技术，避免w变得非常的大

* 在原来的目标函数后加上一个二范数正则化项：目标函数 + 正则部分

* 式子中的 $\lambda$ 就是超参数，他决定了对 $w$ 的限制，起到了平衡的作用

* ![imagedffd8ba200f290b6.png](https://file.moetu.org/images/2020/01/28/imagedffd8ba200f290b6.png)

* 梯度下降使用正则化

  ![imageceec83110ef05091.png](https://file.moetu.org/images/2020/01/28/imageceec83110ef05091.png)



## 模型复杂度和过拟合

* 泛化能力（Generalization Capability）
* 在训练数据上的结果和在测试数据上的结果差别不大，那么这个模型的表现越好
* 仅仅是训练集上表现好在测试集上表现失常就是过拟合

1. 模型本身的选择：LR、SVM、Neural NetWork、deep learning

2. 模型的参数个数：Neural Network的隐藏层神经元数

3. 模型的参数空间选择：正则，在对应的参数空间中选择让模型比较简单的部分

4. 模型拟合过少的样本：样本很少是容易造成过拟合

   ![imageb94c3efcf417c1dd.png](https://file.moetu.org/images/2020/01/28/imageb94c3efcf417c1dd.png)



## 避免过拟合的方法

* 减少模型的参数
* 选择更简单的参数空间



## 正则化介绍

* 正则化其实是对关系式里的权重做调整

  * ![image75a3c2d9761b5179.png](https://file.moetu.org/images/2020/01/28/image75a3c2d9761b5179.png)

* L1-Norm：（sparse solution）

  * 使很多的小值变为零，直接删除这个特征的影响
  * 比如LASSO回归

* L2-Norm：（non-sparse solution）

  * 可以使得其中一些w都很小接近于0，削弱某些特征的影响
  * 越小的参数说明模型越简单，越简单的模型越不容易出现过拟合
  * 比如Ridge回归

* 可以同过绘图的方式来分析，假定 $\theta$ 是二维的：

  ![image4ed6660c9d818a6e.png](https://file.moetu.org/images/2020/01/28/image4ed6660c9d818a6e.png)

* 可以结合L1和L2