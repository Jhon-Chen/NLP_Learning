"""
文本序列化
"""

import numpy as np


class WordSequence:
    UNK_TAG = "UNK"    # 表示位置符号
    PAD_TAG = "TAK"    # 填充符
    UNK = 1
    PAD = 0

    def __init__(self):
        self.dict = {   # 保存词语和对应的数字
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}    # 统计词频
        # 把dict进行翻转
        """
        注：zip()函数用于将可迭代对象作为参数，将对象中对于的元素打包成一个个元组，然后返回由
        这些元组组成的对象，这样的好处是节省了不少的内存。
        至于返回的形式是元组，我们则可以用list()或者dic()来转换输出的类型，使其输出为列表或者字典。
        另外，如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用*操作符，可
        以将其解压为列表。
        """
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def fit(self, sentences):
        """
        接受句子，统计词频
        :param sentences:[str, str, str]
        :return:None
        """
        for word in sentences:
            self.count[word] = self.count.get(word, 0) + 1
            # 所有的句子fit之后，self.count就有了所有词语的词频

    def build_vocab(self, min_count=5, max_count=None, max_features=None):
        """
        根据条件构造  词典
        :param min_count: 最小词频
        :param max_count: 最大词频
        :param max_features: 最大词语数
        :return:
        """
        if min_count is not None:
            self.count = {word: count for word, count in self.count.items() if count > min_count}
        if max_count is not None:
            self.count = {word: count for word, count in self.count.items() if count < max_count}
        if max_features is not None:
            self.count = dict(sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_features])

        for word in self.count:
            self.dict[word] = len(self.dict)  # 每次word对应一个数字

    def transform(self, sentences, max_len):
        """
        把句子转化为数字序列
        :param max_len: 将长度统一
        :param sentences:[str, str, str]
        :return:[int, int, int]
        """
        if len(sentences) < max_len:
            sentences = sentences + [self.PAD_TAG] * (max_len - len(sentences))
        else:
            sentences = sentences[:max_len]
        return [self.dict.get(i, 1) for i in sentences]

    def inverse_transform(self, incidence):
        """
        把数字序列转化为字符
        :param incidence:[int, int, int]
        :return:[str, str, str]
        """
        return [self.inverse_dict.get(i, "UNK") for i in incidence]

    def __len__(self):
        return len(self.dict)

# if __name__ == '__main__':
    # sentence = [["今天", "天气", "很好"], ["今天", "去", "吃", "什么"]]
    # ws = WordSequence()
    # for st in sentence:
    #     ws.fit(st)
    # ws.build_vocab(min_count=0)
    # demo = ws.transform(["很好", "什么", "吃"], max_len=8)
    # print(demo)
    # demo = ws.inverse_transform(demo)
    # print(demo)
















