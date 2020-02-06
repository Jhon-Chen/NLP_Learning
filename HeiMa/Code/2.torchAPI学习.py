import torch
import torch.nn as nn


# y = wx + b
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 自定义的代码
        # self.w = torch.rand([], requires_grad=True)
        # self.b = torch.tensor(0, dtype=torch.float, requires_grad=True)
        # x中有10列[5,10] --- 操作[10,2] + b[2] --- 数据只有2列[5,2]
        self.lr = nn.Linear(1, 10)
        # 假如把这看做是神经网络，那还有以下几层...
        self.lr2 = nn.Linear(10, 20)
        self.lr3 = nn.Linear(20, 1)

    def forward(self, x): # 完成一次向前计算
        # y_predict = x * self.w + self.b
        # return y_predict
        # return self.lr(x)
        # 神经网络的一个向前计算过程
        out1 = self.lr(x)
        out2 = self.lr2(out1)
        out = self .lr3(out2)
        return out


# 调用模型
if __name__ == '__main__':
    # model = MyModel()，要注意下边的这个数据的类型，在计算w*x+b时如果数据的类型不一样是无法进行计算的
    # y_pred = MyModel()(torch.FloatTensor([10]))
    # print(y_pred)
    model = MyModel()
    print(model.parameters())
    for i in model.parameters():
        print(i)
        print("*"*100)