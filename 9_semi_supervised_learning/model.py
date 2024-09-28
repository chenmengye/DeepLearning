import torch
import torchvision
from torch import nn
from config import HP


class WideResnet50_2(nn.Module):
    def __init__(self):
        super(WideResnet50_2, self).__init__()
        resnet = torchvision.models.wide_resnet50_2(pretrained=False)
        last_fc_dim = resnet.fc.in_features # defaut imagenet, 1000
        fc = nn.Linear(in_features=last_fc_dim, out_features=HP.classes_num)
        resnet.fc = fc
        self.wideresnet4cifar10 = resnet

    def forward(self, input_x):
        return self.wideresnet4cifar10(input_x)


if __name__ == '__main__':
    model = WideResnet50_2()
    ret = model(torch.randn(size=(7, 3, 32, 32)))
    print(ret.size())

