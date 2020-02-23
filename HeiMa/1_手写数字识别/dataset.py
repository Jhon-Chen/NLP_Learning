"""
准备数据集
"""

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
import conf


def mnist_dataset(train):  # 准备mnist的dataset
    func = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    # 1.准备Mnist数据集
    return MNIST(root="C:\\Users\\Administrator\\Git\\NLP_Learning\\HeiMa\\1_手写数字识别", train=train, download=False, transform=func)


def get_dataloader(train=True):
    mnist = mnist_dataset(train)
    batch_size = conf.train_batch_size if train else conf.test_batch_size
    return DataLoader(mnist, batch_size=batch_size, shuffle=True)


if __name__ == '__main__':
    for (images, labels) in get_dataloader():
        print(images.size())
        print(labels.size())
        break

