# day_2



### 梯度下降

1. 梯度：向量，值是导数，方向是变化最快的放下
2. 导数的计算
3. 更新参数w

### 反向传播

1. 链式法则
2. 计算图
3. 反向传播是链式求导的一种算法

### PyTorch实现线性回归

1. tensor中的`require_grad`参数

   * 设置为True，表示会记录该tensor的计算过程

2. tensor中的`grad_fn`属性

   * 用来保存计算的过程

3. 可以在创建tensor是使require_grad为True，或者`tensor.require_grad_(True)`

4. 还可以使用`with torch.no_grad():...` 使得以下的操作都不被记录在`grad_fn`中

5. **反向传播**

   * 在输出为一个标量的情况下，我们可以调用输出`tensor`中的`backward()`方法，但是在数据是一个向量的时候，调用`backward()`还需要传入其他的参数。导数保存在`tensor.grad`中
   * **默认情况下，要多次进行反向传播求梯度要设置`backward(retain_graph=True)`，并且它的梯度结果会在`grad`中累加**
   * `loss.backward()`就是根据损失函数，对参数`requires_grad=True`去计算他的梯度并且把它累加保存到`x.grad`中，此时还未更新其梯度。

6. `tensor.data`

   * 一般要对tensor的值进行操作，就会使用`tensor.data`，相当于对tensor中的值进行引用
   * 同时，在tensor中有`grad_fn`时，不可以直接把tensor转换成numpy对象
   * 需使用`tensor.detach()`，表示从tensor中取值然后才能转换成numpy（data也可以）

7. 实现线性回归的逻辑

   1. 准备数据
   2. 初始化参数，**进入循环（参数的梯度置为0）**，计算预测值
   3. 计算loss，`loss.backwarda()`计算梯度
   4. 更新参数

8. PyTorch通过API完成模型和训练

   1. **API：**

      * **`nn.Module`构造模型：**

        > `nn.Module`是`torch.nn`提供的一个类，是PyTorch中我们自定义网络的一个基类，在这个类中定义了很多有用的方法可以让我们在继承这个类定义网络的时候非常简单
        >
        > 当我们在自定义网络的时候，有两个方法需要特别注意：
        >
        > 1. `__init__` 需要调用`super`方法，继承父类的属性和方法
        > 2. `farward`方法必须实现，用来定义我们的网络的向前计算过程
        >
        > 用前面的`y=wx+b`模型举例如下：

        1. `init`：自定义的方法实现的位置
        2. `forward`：完成一次向前计算的过程
        3. `nn.Linear`为torch预定义好的现象模型，也被称为**全链接层**，传入的参数为输入的数量，输出的数量`(in_features, out_features)`，是不算`(batch_size)`的列数
        4. `nn.Module`定义了`__call__`方法，实现的就是调用`forward`方法，即`Lr`的实例，能够直接被传入参数调用，实际上调用的是`forward`方法并传入参数

      * **optimizer优化器类：**

        > 优化器（optimizer），可以理解为torch为我们封装的用来进行数据更新参数的方法，比如常见的随机梯度下降（`stochastic gradient descent, SGD`）
        >
        > 优化器类都是由`torch.optim`提供的，例如
        >
        > 1. `torch.optim.SGD(参数，学习率)`
        > 2. `torch.optom.Adam(参数，学习率)`

        注意：

        1. 参数可以使用`model.parameters()`来获取，获取模型中所有`requires_grad=True`的参数
        2. 优化类的使用方法
           1. 实例化，`Aadm(model.parameters(),lr)`
           2. 所有参数的梯度，将其置为零
           3. 反向传播计算梯度，`loss.backward()`
           4. 更新参数值，`optimizer.setp()`

      * **损失函数：**

        > 1. 均方误差：`nn.MESLoss()`，常用于回归问题
        > 2. 交叉熵损失：`nn.CrossEntropyLoss`，常用于分类问题

        流程：

        1. 实例化模型
        2. 实例化优化器类  
        3. 实例化损失函数
        4. 进入循环：
           1. 梯度置为0
           2. 调用模型得到预测值
           3. 调用loss函数，得到损失
           4. `loss.backward()`进行梯度计算
           5. `optimizer.step()`

      * **注意点：**

        1. `model.eval`() 表示把模型置为评估模式
           * `model.trainingg=False`

   2. **在GPU上运行代码**

      当模型太大，或者参数太多的情况下，为了加快训练的速度，经常会使用GPU来进行训练，此时我们的代码需要稍作调整:

      1. 判断GPU是否可用`torch.cuda.is_available()`

         ```python
         torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
         ```

      2. 把模型参数和input数据转换为cuda支持的类型

         ```python
         model.to(device)
         x_true.to(device)
         ```

      3. 在GPU上计算结果也为cuda的数据类型，需要转化为numpy或者torch的cpu的tensor类型

         ```python
         predict = predict.cpu().detach().numpy()
         ```

         `detach()`的效果与`data()`类似，但是`detach`是引用。



### 常见的优化算法介绍

1. 梯度下降算法（batch gradient descent BGD）

   每次迭代都需要把所有的样本都送入，这样的好处是每次迭代都顾及了全部的样本，做的是全局优化，但是有可能达到局部最优。

2. 随机梯度下降法（Stochastic gradient descent SGD）

   针对梯度下降算法训练速度过慢的缺点，提出了随机梯度下降算法，随机梯度下降算法是从样本中随机抽出一组，训练后按梯度更新一次，然后再抽取一组，再更新一次，在样本量及其大的情况下，可能不用训练完所有的样本就可以获得一个损失值在可接受范围之内的模型了。

   `torch`中的API为：`torch.optim.SGD()`

3. 小批量梯度下降（Mini-batch gradient descent MBGD）

   SGD相对来说要快很多，但是也存在问题，由于单个样本的训练可能会带来很多的噪声，使得SGD并不是每次迭代都向着整体最优化方向，因此在刚开始可能收敛的很快，但是训练一段时间以后就会变得很慢。在此基础上又提出了小批量梯度下降法，它是每次从样本中随机抽取一小批进行训练，而不是一组，这样即保证了效果又保证了速度。

4. 动量法

   mini-batch SGD算法虽然能够带来很好的训练速度，但是在到达最优点的时候并不能够总是真正的到达最优点，而是在最优点附近徘徊。

   另一个缺点就是mini-batch SGD需要我们挑选一个合适的学习率，当我们采用小的学习率的时候，会导致网络在训练的时候收敛太慢。当我们采用大的学习率的时候，会导致在训练过程中优化的幅度跳过函数的范围，也就是可能跳过最优点。我们所希望的仅仅是网络在优化的时候网络的损失函数有一个很好的收敛速度同时又不至于摆动幅度太大。

   所以Momentum优化器刚好可以解决我们所面临的问题，它主要是基于梯度的移动指数加权平均，对网络的梯度进行平滑处理的，让梯度的摆动幅度变得更小。

   ![image8d0ce59bba73d57e.png](https://file.moetu.org/images/2020/02/06/image8d0ce59bba73d57e.png)

5. AdaGrad

   AdaGrad算法就是将每一个参数的每一次迭代的梯度取平方累加后再开方，用全局学习率除以这个数，作为学习率的动态更新，从而达到**自适应学习率**的效果。

   ![image9c9fc8f934b4d55c.png](https://file.moetu.org/images/2020/02/06/image9c9fc8f934b4d55c.png)

6. RMSProp

   Momentum优化算法中，虽然初步解决了优化汇总摆动幅度大的问题，为了进一步优化算是函数在更新中存在摆动幅度过大的问题，并且进一步加快函数的收敛速度，RMSProp算法对参数梯度使用来了平方加权平均数。

   ![imagecf640098180e896f.png](https://file.moetu.org/images/2020/02/06/imagecf640098180e896f.png)

7. Adam

   Adam（Adaptive Moment Estimation）算法是将Momentum算法和RMSProp算法结合起来的一种算法，能够达到防止梯度的摆幅过大，同时还能够提高收敛速度。

   ![imagec51d0f244ee7bfac.png](https://file.moetu.org/images/2020/02/06/imagec51d0f244ee7bfac.png)

   `torch`中的API为：`torch.optim.Adam()`



