# Project_01

[toc]



## Python range()函数

![image-20191230203843944](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20191230203843944.png)



## 分词原理

* 向前最大匹配
  * 最大匹配有一个长度max_len，比如max_len = 5，那么从前往后，每次减少结尾的单词使得他出现一个在词库中出现的词为止
* 向后最大匹配
  * 与向前最大匹配相同，只是方向相反
* 最大匹配是基于匹配规则的方法
* LM，HMM，CRF是基于概率统计的方法
* 分词可以认为是已经解决的问题



## 分词-语言模型

* Ungram Language Model：假定所有的词都是独立的，那这个句子就是他们概率的乘积（不合理）
* 一般语言模型会考虑上下文



## 分词-维特比算法（重点）

![image-20200101170608866](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200101170608866.png)

拿到维特比算法考虑先画一个图：

![image-20200101170646209](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200101170646209.png)





## Spell Correction（拼写错误纠正）

* 错别字
* 不是错别字，但是用的不合适

从用户输入（input）到候选（candidates）到**编辑距离（edit distance）重点！！DP算法**

有三种错误的情况：

1. insert（插入）
2. delete（删除）
3. replace （替代）