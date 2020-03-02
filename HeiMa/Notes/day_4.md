# day_4

[toc]



## 案例_文本情感分类



### 案例介绍

对前面的word embedding这种常用的文本向量化方法进行巩固。

现在有一个经典的数据集`IMDB`地址：`http:/ai.stanford.edu/~amaas/data/sentiment/`，这是一份包含了5万条流行电影评论数据，其中训练集25000条，测试集25000条。下图上边是名称，其中名称包含两部分，分别是序号和情感评分，（1-4为neg，5-10为pos），右边为评论内容

![image3bf30bdfc5d704ed.png](https://file.moetu.org/images/2020/02/27/image3bf30bdfc5d704ed.png)

根据上述的样本，需要使用pytorch完成模型，实现对评论情感进行预测

**思路分析**

首先可以把上述问题定义为分类问题，情感评分分为1-10，10个类别（也可以理解为回归问题，这里当做分类问题考虑）。那么根据之前的经验，我们的大致流程如下：

1. 准备数据集
2. 构建模型
3. 模型训练
4. 模型评估

知道思路以后，那么我们一步步来完成上述步骤

### 准备数据集

准备数据集和之前的方法一样，实例化dataset，准备dataloader，最终我们的数据可以处理成如下格式。

![image7b90126ff0b4cf39.png](https://file.moetu.org/images/2020/02/27/image7b90126ff0b4cf39.png)

其中有两点需要注意：

1. 如何完成基础打Dataset的构建
2. 每个batch中文件的长度不一致的问题如何解决
3. 每个batch中的文本如何转化为数字序列

#### 基础Dataset的准备

* `__init__`定义读取文件
* `__getitem__`定义下标去元素方法
* `__len__`返回取元素方法

#### dataloader的准备

* 在配置文件中设置好测试集和训练集的batch_size并调用

* 要**注意Dataloader中的一个参数`collate_fn`**

  * `collate_fn`的默认值为torch自定义的`default_collate`，而`collate_fn`的作用就是对每个batch进行处理，而默认的`default_collate`处理出错。（他会把不同的review合并到一起统计）

    ![image8a0ebc8e0b9651ab.png](https://file.moetu.org/images/2020/02/28/image8a0ebc8e0b9651ab.png)

  * 解决方案：自定义一个collate_fn

    ```python
    def collate_fn(batch):
        """
        对batch数据进行处理
        :param batch:[getitem的结果，getitem的结果，...]
        :return: 元组
        """
        review, labels = zip(*batch)
        return review, labels
    ```



#### 文本序列化

>在介绍word embedding的时候，我们说过，不会直接把文本转化为向量，而是线转化为数字，再把数字转化为向量，那么这个过程应该如何实现呢？

这里我们考虑把文本中的每次**词语和其对应的数字用字典保存，**同时实现方法把**句子通过字典映射为包含数字的列表**。

实现文本序列化之前要考虑一下几点：

1. 如何使用字典把词语和数字进行对应
2. 不同词语出现的次数不尽相同，是否需要对高频或者低频词语进行过滤，以及总的词语数量是否需要进行限制
3. 得到词典之后，如何把句子转化为数字序列，如何把数字序列转化为句子
4. 不同句子长度不同，每个batch的句子如何构造成相同的长度（可以对短句进行填充）
5. 对于新出现的词语没有在字典中的怎么办（使用特殊字符替代）

**思路分析**：

1. 对所有句子进行分词
2. 词语存入字典，根据次数对词语进行过滤并统计次数
3. 实现文本转数字序列的方法
4. 实现数字序列转文本的方法

**word sequence的保存：**

在`main`文件中导入`pickle`

`pockle.dump(ws, open("路径"))` 保存对应的文件。

### 构建模型

这里我们只练习使用`word embedding`，所以模型只有一层，即：

1. 数据经过`word embedding`
2. 数据通过全连接层返回结果，计算`log_softmax`