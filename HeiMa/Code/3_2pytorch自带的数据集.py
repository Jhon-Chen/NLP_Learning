from torchvision.datasets import MNIST
import torchvision
from torch.utils.data import DataLoader

func = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))])


# 1. 准备Mnist数据集
mnist = MNIST(root="/Data/mnist", train=True, download=False, transform=func)
# print(mnist)
# print(mnist[0])
# print(len(mnist))
# print(mnist[0][0].show()) 打开图片

# 2. 准备数据加载器
dataloader = DataLoader(dataset=mnist, batch_size=2, shuffle=True, num_workers=2)


if __name__ == '__main__':
    for idx, (images, labels) in enumerate(dataloader):
        print(idx)
        print(images)  # [2, 1, 28, 28]，第一个2是因为batch_size
        print(labels)  # 2
        break

