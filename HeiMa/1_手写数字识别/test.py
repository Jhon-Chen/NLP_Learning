"""
进行模型的评估
"""

import numpy as np
from dataset import get_dataloader
from models import MnistModel
from torch import optim
import torch.nn.functional as F
import conf
from tqdm import tqdm
import torch
import os

# 1. 实例化模型，优化器，损失函数
model = MnistModel().to(conf.device)


def eval():
    # 模型的加载，先判断是否存在
    if os.path.exists("./model_save/model.pkl"):
        model.load_state_dict(torch.load("./model_save/model.pkl"))
    test_dataloader = get_dataloader(train=False)
    # bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    total_loss = []
    total_acc = []
    with torch.no_grad():
        for input, target in test_dataloader:
            input = input.to(conf.device)
            target = target.to(conf.device)

            # 计算得到预测值
            output = model(input)
            # 得到损失
            loss = F.nll_loss(output, target)
            # 反向传播得到损失
            total_loss.append(loss.item())
            # 计算准确率
            # 计算预测值
            pred = output.max(dim=-1)[-1]
            total_acc.append(pred.eq(target).float().mean().item())
    print("test loss:{}, test acc:{}".format(np.mean(total_loss), np.mean(total_acc)))


if __name__ == '__main__':
    # for i in range(5):
    #     train(i)
    eval()