import torch
from torch import nn
from config import HP
import torchvision

# 1. Pretrained ResNet 34
# 2. delete and modify fc


class MoocTLNet(nn.Module): # mooc transfer learning net
    def __init__(self):
        super(MoocTLNet, self).__init__()
        self.model = torchvision.models.resnet34(pretrained=True)
        if HP.if_conv_frozen:
            for k, v in self.model.named_parameters():
                v. requires_grad = False
        resnet_fc_dim = self.model.fc.in_features
        new_fc_layer = nn.Linear(resnet_fc_dim, out_features=HP.classes_num) # new fc layer
        self.model.fc = new_fc_layer

    def forward(self, input_x):
        return self.model(input_x)


if __name__ == '__main__':
    x = torch.randn(size=(7, 3, 112, 112))
    model = MoocTLNet()
    output = model(x)
    print(output.size())
    for k, v in model.named_parameters():
        if v.requires_grad:
            print(k)



