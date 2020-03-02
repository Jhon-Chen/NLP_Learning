"""
准备数据集
"""

import os
from torch.utils.data import DataLoader, Dataset
import config
from utils import tokenlize
import torch

class ImdbDataset(Dataset):
    def __init__(self, train=True):
        super(ImdbDataset, self).__init__()
        data_path = r"C:\Users\Administrator\Git\NLP_Learning\HeiMa\2_文本情感分类\data"
        data_path += r"\train" if train else r"\test"
        self.total_path = []  # 保存所有的文件路径
        for temp_path in [r"\pos", r"\neg"]:
            cur_path = data_path + temp_path
            """
            注：os.path.join(a, b) 表示将路径自动组合，返回 a/b
                os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
            """
            self.total_path += [os.path.join(cur_path, i) for i in os.listdir(cur_path) if i.endswith(".txt")]

    def __getitem__(self, index):
        file = self.total_path[index]
        reviews = tokenlize(open(file, encoding="utf-8").read())
        labels = int(file.split("_")[-1].split(".")[0])
        labels = 0 if labels < 5 else 1
        return reviews, labels

    def __len__(self):
        return len(self.total_path)


def collate_fn(batch):
    """
    对batch数据进行处理, 这是get_dataloader里的一个处理方法
    :param batch:[getitem的结果，getitem的结果，getitem的结果]
    :return: 元组
    """
    reviews, labels = zip(*batch)
    reviews = [config.ws.transform(i, max_len=config.max_len) for i in reviews]
    reviews = torch.LongTensor(reviews)
    labels = torch.LongTensor(labels)
    return reviews, labels


def get_dataloader(train=True):
    dataset = ImdbDataset(train)
    batch_size = config.train_batch_size if train else config.test_batch_size
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


if __name__ == '__main__':
    # dataset = ImdbDataset()
    # print(dataset[0])
    for idx, (review, label) in enumerate(get_dataloader(train=True)):
        print(idx)
        print(review)
        print(label)
        break
