import torch
import matplotlib.pyplot as plt

learning_rate = 0.02

# 1. 准备数据  y = 3x +0.8
x = torch.randn([500, 1])
y_true = 3 * x + 0.8

# 2. 计算预测值 y_pred = x * w + b
w = torch.rand([], requires_grad=True)
b = torch.tensor(0, dtype=torch.float, requires_grad=True)

for k in range(500):
    # 由于每一次grad的值会累加，所以我们要判断当grad的值不为None时就把它置为0
    for i in [w, b]:
        if i.grad is not None:
            i.grad.data.zero_()
    y_pred = x * w + b

    # 3. 计算损失，把参数的梯度置为0，进行反向传播
    # 注：pow()是做幂次运算，回归任务的损失函数可以使用均方误差来计算
    loss = (y_true - y_pred).pow(2).mean()

    loss.backward()
    # 反向传播以后可以得到w和b的梯度
    # 3. 更新参数
    w.data = w.data - learning_rate * w.grad
    b.data = b.data -learning_rate * b.grad
    print(k, loss.item())
print(w, b)

plt.figure(figsize=(20, 8))
plt.scatter(x.numpy(), y_true.numpy())

y_pred = x * w + b
plt.plot(x.numpy(), y_pred.detach().numpy(), c='red')
plt.show()
