import torch
import torch.nn as nn
from torch import optim
import time


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lr = nn.Linear(1, 1)

    def forward(self, x): # x [500, 1] --- y_pred [500,1]
        return self.lr(x)


# 0.准备数据
x = torch.rand([500,1])
y_true = 3*x + 0.8

# 1.实例化模型
model = MyModel()
# 2.实例化优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)
# 3.实例化损失函数
loss_fn = nn.MSELoss()

t0 = time.time()
# 4.开始循环
for i in range(50000):
    # 梯度置为0
    optimizer.zero_grad()
    # 调用模型获得预测值
    y_pred = model(x)
    # 通过损失函数，计算得到的损失
    loss = loss_fn(y_pred, y_true)
    # 进行反向传播计算梯度
    loss.backward()
    # 进行参数更新
    optimizer.step()

    # 打印部分数据
    if i % 10 == 0:
        print(i, loss.item())
print(time.time() - t0)