import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SmsDataset(Dataset):
    def __init__(self):
        self.file_path = "/Code/Data\\SMSSpamCollection.txt"
        self.lines = open(self.file_path, encoding='utf-8').readlines()

    def __getitem__(self, index):
        line = self.lines[index].strip()
        label = line.split('\t')[0]
        content = line.split('\t')[1]
        return label, content

    def __len__(self):
        return len(self.lines)


# 使用DataLoader
# 先实例化Dataset
sms_dataset = SmsDataset()
# print(sms_dataset[0])
dataloader = DataLoader(dataset=sms_dataset, batch_size=5, shuffle=True)

if __name__ == '__main__':
    for idx, (labels, contents) in enumerate(dataloader):
        print(idx)
        print(labels)
        print(contents)
        break
    print(len(sms_dataset))
    print(len(dataloader))

