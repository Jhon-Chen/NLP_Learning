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

## Noisy Channel Model

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

