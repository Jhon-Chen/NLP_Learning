"""
构建模型
"""
import torch.nn as nn
import config
import torch.nn.functional as F


class ImdbMode(nn.Module):
    def __init__(self):
        super(ImdbMode, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.ws), embedding_dim=200, padding_idx=config.ws.PAD)
        # padding_idx 表示那个符号是填充符
        self.fc = nn.Linear(config.max_len*200, 2)

    def forward(self, input):
        """
        :type input: Tensor
        :param input: [batch_size, max_len]
        :return:
        """
        # input embed :[batch_size, max_len, embedding_dim]
        input_embed = self.embedding(input)

        # 变形
        input_embed_viewed = input_embed.view(input_embed.size(0), -1)

        # 全连接
        out = self.fc(input_embed_viewed)
        return F.log_softmax(out, dim=-1)