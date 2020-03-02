"""
手动实现线性回归练习
"""

import torch

learning_rate = 0.01

# 准备数据
x = torch.rand(10, 3)

w_true = torch.tensor([[1], [2], [3]], dtype=torch.float)  # 形状[3, 1] ---> 计算后为 [10, 1]
b_true = torch.tensor([10], dtype=torch.float)  # 注意，b就是[1, 1]
y_true = torch.matmul(x, w_true) + b_true

# 计算预测值
w = torch.rand([3, 1], requires_grad=True)
b = torch.rand([1, 1], requires_grad=True)


def loss_fn(y_true, y_predict):
    loss = (y_true - y_predict).pow(2).mean()
    loss.backward()
    return loss.item()


def optimize(learning_rate):
    w.data -= learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data


for k in range(5000):
    for i in [w, b]:
        if i.grad is not None:
            i.grad.data.zero_()
    y_predict = torch.matmul(x, w) + b

    loss = loss_fn(y_predict, y_true)

    if i % 500 == 0:
        print(i, loss)
    optimize(learning_rate)

print("w", w)
print("b", b)

