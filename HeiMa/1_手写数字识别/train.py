""""
进行模型的训练
"""

from dataset import get_dataloader
from models import MnistModel
from torch import optim
import torch.nn.functional as F
import conf
from tqdm import tqdm
import torch
import os
from test import eval

# 1. 实例化模型，优化器，损失函数
model = MnistModel().to(conf.device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 模型的加载，先判断是否存在
# if os.path.exists("./model_save/model.pkl"):
#     model.load_state_dict(torch.load("./model_save/model.pkl"))
#     optimizer.load_state_dict(torch.load("./model_save/optimizer.pkl"))


# 2. 进入循环，进行训练
def train(epoch):
    train_dataloader = get_dataloader(train=True)
    bar = tqdm(train_dataloader, total=len(train_dataloader))
    for idx, (input, target) in enumerate(train_dataloader):
        input = input.to(conf.device)
        target = target.to(conf.device)
        # 把梯度置为零
        optimizer.zero_grad()
        # 计算得到预测值
        output = model(input)
        # 得到损失
        loss = F.nll_loss(output, target)
        # 反向传播得到损失
        loss.backward()
        # 参数更新
        optimizer.step()
        # 打印数据
        if idx % 10 == 0:
            # print("epoch:{}  idx:{}  loss:{:.6f}".format(epoch, idx, loss.item()))
            bar.set_description("epoch:{}, idx:{}, loss:{:.6f}".format(epoch, idx, loss.item()))
            # 保存模型
            torch.save(model.state_dict(), "./model_save/model.pkl")
            torch.save(optimizer.state_dict(), "./model_save/optimizer.pkl")


if __name__ == '__main__':
    for i in range(5):
        train(i)
        eval()