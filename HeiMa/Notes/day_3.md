# day_3

[toc]



## PyTorch中的数据记载

### 模型中使用数据加载器的目的

在深度学习中，数据量通常是非常多，非常大对的，如此大量的数据，不可能一次性的在模型中进行向前的计算和反向传播，经常我们会对整个数据进行随机的打乱顺序，把数据处理成一个个的batch，同时还会对数据进行预处理。

`epoch`：那所有的数据训练一次

`batch`：数据打乱顺序，组成一波一波的数据，批处理



### 补充基础回顾（Python类）

1. `__init__`方法

   Python的类里提供的，两个下划线开始结尾的方法，就是魔方方法；如果类里面没有写`__init__`方法，Python会自动创建，但是不执行任何操作；如果为了能够完成自己想要的功能，可以自己定`__init__`方法；所以捂脸一个类里是否编写`__init__`方法，它都**一定存在**。

   此方法一般用来做**变量初始化**或者**赋值**操作，在类实例化对象的时候，会被自动调用。勒种其他的方法叫做实例方法。其中的`self`参数不需要开发者传递，Python解释器会自动把当前对象引用传递过去。

2. 在类内部获取属性和实例方法，通过`self`获取；在类外部获取属性和方法，通过对象名获取；如果一个类有多个对象，每个对象的属性是各自保存的，都有各自独立的地址；但是实例方法是所有对象共享的，只占用一份内存空间，类会通过`self`来判断是哪个对象调用了实例方法。

3. `__str__`方法

   这也是一个魔方方法，用来显示信息。该方法需要`return`一个数据，并且只有`self`一个参数，当在类的外部`print(对象)`时，就会打印`return`的内容。



### 数据集类

#### Dataset基类介绍

在torch中提供了数据集的基类`torch.utils.data.Dataset`，继承这个基类，我们能够非常快速的实现对数据的加载。

* 两个重要方法：

  1. `__getitem__(index)`：能够对实例进行索引

  2. `__len__`：len(实例)就是调用词方法返回长度
  3. 另外，我们会在`__init__`中初始化文件的读取

#### 迭代数据集

使用上述的方法能够进行数据的读取，但是其中还有很多内容没有实现：

* 批处理数据（Batching the data）
* 打乱数据（Shuffing the data）
* 使用多线程`multiprocessing`并行加载数据

在pytorch中的`torch.utils.data.DataLoader`提供了上述的所用方法

1. `torch.utils.data.DataLoader`

   ```python
   data_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=2)
   ```

   其中参数的含义：

   1. `dataset`：提前定义的dataset实例
   2. `batch_size`：传入数据的batch的大小，常用128，256等等
   3. `shuffle`：bool类型，表示是否在每次获取数据的时候提前打乱数据
   4. `num_workers`：加载数据的线程数

   

   #### Pytorch自带的数据集

   pytorch中自带的数据集由两个上层api提供，分别是`torchvision`和`torchtext`，其中：

   1. `torchvision`提供了对图片数据处理相关的api数据
      * 数据位置：`torchvision.datasets`，例如：`torchvision.datasets.MNIST`（手写数字图片数据）
   2. `torchtext`提供了对文本数据处理相关的API和数据
      * 数据位置：`torchtext.datasets`，例如：`torchtext.datasets.IMOB`（电影评论文本数据）

   

   ### 手写数字识别的思路

   流程：

   	1. 准备，通过dataset和DataLoader的准备
    	2. 模型的构建
    	3. 模型训练，模型保存和加载
    	4. 模型的评估

   

   #### 1.准备Mnist数据：

   * `torchvision.transforms`的图形数据处理方法

     1. `torchvision.transforms.ToTensor`

        把一个取值范围是`[0, 255]`的`PIL.Image`或者`shape`为`(H, W, C)`的`numpy.ndarray`，转换成形状为`[C, H, W]`。

        其中`(C, H, W)`的意思是`(高, 宽, 通道数)`，黑白图片的通道数只有1，其中每个像素点的取值为`[0, 255]`,彩色图片的通道数为`(R, G, B)`，每个像素点的取值为`[0, 255]`，三个通道的颜色相互叠加形成了各种颜色。

     2. `torchvision.transforms.Normalize(mean, std)`

        给定均值：`mean`，`shape`和图片的通道数相同（指的是每个通道的均值），方差：`std`，和图片的通道数相同，将会把tensor**规范化**处理。

        即：`Normalized_image=(image-mean)/std`

        注意，在api：`Normalize`中并没有帮我们计算`std`和`mean`，所以需要我们手动计算。

        1. 当`mean`为全部数据的均值，`std`为全部数据的`std`是，才是进行了标准化。

        2. 如果`mean(x)`不是全部数据的mean的时候，`std(y)`也不是的时候，`Normalize`后的数据分布满足下面的关系：

           **![image2e6110d3093e1387.png](https://file.moetu.org/images/2020/02/08/image2e6110d3093e1387.png)**

     3. `torchvision.transforms.Compose(transforms)`

        将多个`transform`组合起来使用。例如：

        ```
        transforms.Compose([
        	torchvision.transforms.ToTensor(), # 先转化为Tensor
        	torchvision.transforms.Normalize(mean, std) # 再进行正则化
        ])
        ```

   #### 2. 构建模型

   * 补充：**全连接层**：当前一层的神经元和前一层的神经元互相链接，其核心操作就是 $y=wx$，即矩阵的乘法，实现对前一层的数据的变换。
   * 模型的构建使用了一个三层的神经网络，其中包括两个全连接层和一个输出层，第一个全连接层会经过计划函数的处理，将处理后的结果交给下一个全连接层，进行变换后输出结果，那么在这个模型中有两个地方需要注意：：
     1. 激活函数如何使用
     2. 每一层数据的形状
     3. 模型的损失函数

   * **激活函数的使用**

     常用的激活函数为Relu激活函数，它的使用非常简单，Relu激活函数由`import torch.nn.functional as F`提供，`F.rule(x)`即可对`x`进行处理。

   #### 3.模型的损失函数

   首先，我们需要明确，当前我们手写字体识别的问题是一个多分类问题，所谓多分类对比的是之前学习的2分类。回顾之前的课程，我们在逻辑回归中，我们使用`sigmoid`进行计算对数似然损失，来定义我们的2分类损失。

   * 在2分类中我们有正类和负类，正类的概率为![imagee89b04cc818fe229.png](https://file.moetu.org/images/2020/02/08/imagee89b04cc818fe229.png)，那么负类就是$1-P(x)$。
   * 将这个结果进行计算对数似然损失$-\sum ylog(P(x))$就可以得到最终的算是

   *那么在多分类中应该如何做呢？*

   * 多分类与二分类中唯一的区别就是我们不能够在使用`sigmoid`函数类计算当前样本属于某个类别的概率，而应该改用`softmax`函数。

   * `softmax`和`sigmoid`的区别在于我们需要去计算样本数据每个类别的概率，需要计算多次，而`sigmoid`只需要计算一次，`softmax`的公式如下：

     ![imagef8575114d90c9d1a.png](https://file.moetu.org/images/2020/02/08/imagef8575114d90c9d1a.png)

     例如下图：
     ![imageac9e63e5aea75c28.png](https://file.moetu.org/images/2020/02/19/imageac9e63e5aea75c28.png)

     和前面的2分类的损失一样，多分类的损失只需要再把这个结果进行对数似然损失的计算即可：
     ![image1402f6ecfe01e746.png](https://file.moetu.org/images/2020/02/19/image1402f6ecfe01e746.png)

     最后，会计算每个样本的算是，即上式的平均值

     我们把softmax概率传入对数似然损失得到的损失函数称为**交叉熵损失**

     在Pytorch中有两种方法实现交叉熵损失：

     * ```python
       softmax(out)   # 指数/指数和
       criterion = nn.CrossEntropyLoss()
       loss = criterion(input, target)
       ```

     * ```python
       # 1. 对输出值计算softmax并取对数
       output = F.log_softmax(x, dim=-1)   # log(P)
       # 2. 使用torch中带权损失 计算交叉熵损失
       loss = F.nll_loss(output, target)   # -\sum y*log(P)
       ```

     ![image5ab240bb1b50df5b.png](https://file.moetu.org/images/2020/02/19/image5ab240bb1b50df5b.png)



#### 4.模型的训练

训练的流程：

1. 遍历dataloader
2. tqdm(可迭代对象， total=迭代总次数)



#### 5.模型的保存和加载

```python
# 模型的加载，先判断是否存在
if os.path.exists("./model_save/model.pkl"):
    model.load_state_dict(torch.load("./model_save/model.pkl"))
  optimizer.load_state_dict(torch.load("./model_save/optimizer.pkl"))
# 保存模型
    torch.save(model.state_dict(), "./model_save/model.pkl")
    torch.save(optimizer.state_dict(), "./model_save/optimizer.pkl") 
```



#### 6.模型的评估

评估的过程和训练的过程相似，但是：

1. 不需要计算梯度
2. 需要收集损失和准确率，用来计算计算平均损失和平均准确率
3. 损失的计算和训练时候损失的计算方法相同
4. 准确率的计算：
   * 模型的输出为 [batch_size, 10] 的形状
   * 其中最大值的位置就是其预测的目标值（预测值进行过softmax后为概率，softmax中分母都是相同，分子越大，概率越大）
   * 最大值的位置获取的方法可以使用`torch.max`返回最大值和最大值的位置
   * 返回最大值的位置后，和真实值`([batch_size])`进行对比，相同表示预测成功



## 循环神经网络和自然语言处理介绍



### 文本的`tokenization` （分词）

1. 概念和工具的介绍

   `tokenizaiton`就是通常所说的分词，分出的每一个词语我们把它称为`token`。

   常见的分词工具很多，比如：

   * `jieba`分词
   * 清华大学的分词工具THULAC

2. 中英文的分词方法

   * 把句子转化为词语
   * 把句子转化为单字

### N-gram模型

前面提到，句子可以用单个字，词来表示，但是有的时候，我们也可以用2个，3个或者多个词来表示。

N-gram是一组一组的词语，其中的N表示能够被一起使用的词的数量。

### 向量化

因为文本不能直接被模型计算，所以需要将其转化为向量

把文本转化为向量有两种方法：

1. 转化为one-hot编码
2. word emebdding

**word emebdding**

word emebdding是深度学习中表示文本常用的一种方法。和one-hot编码不同，word embedding 使用了浮点型的稠密矩阵来表示token。根据词典的大小，我们的向量通常使用不同的维度，例如100，256，300等。其中向量的每一个值是一个参数，其初始值是随机生成的，之后会在训练中进行学习获得。

**word emebdding API**

`torch.nn.Embedding(num_embeddings, embedding_dim)`

里面的值都是参数，都会在后续通过训练得到。

参数介绍：

* `num_embedding`：词典的大小
* `embedding_dim`：`embeddimg`的维度，每一个词的维度

使用方法：

```python
embedding = nn.Embedding(vocab_size, 300) # 实例化
input_embeded = embedding(input)  # 进行embedding操作
```

**数据形状的变化**

思考：每个batch中的每个句子有10个词语，经过形状为word embedding(10, 4)之后，原来的句子会变成什么形状？

每个词语用长度为4的向量表示，所以，最终句子会变为`[batch_size, 10, 4]`的形状。增加了一个维度，这个维度是embedding。