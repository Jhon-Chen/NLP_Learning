"""
主程序运行部分
"""

from word_sequence import WordSequence
from dataset import get_dataloader
from tqdm import tqdm
import pickle

if __name__ == '__main__':
    ws = WordSequence()
    for review, label in tqdm(get_dataloader(train=True), total=len(get_dataloader(train=True))):
        # print(review)
        for sentence in review:
            ws.fit(sentence)

    for review, label in tqdm(get_dataloader(train=False), total=len(get_dataloader(train=False))):
        for sentence in review:
            ws.fit(sentence)

    ws.build_vocab()
    print(len(ws))
    pickle.dump(ws, open("./models/ws.pkl", 'wb'))