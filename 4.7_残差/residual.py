import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_feature):
        super(ResidualBlock, self).__init__()
        hidden_chs = 128
        self.weight_layer_1 = nn.Conv2d(in_channels=in_feature, out_channels=hidden_chs, kernel_size=(3, 3), padding=(3//2, 3//2))
        self.weight_layer_2 = nn.Conv2d(in_channels=hidden_chs, out_channels=in_feature, kernel_size=(5, 5), padding=(5//2, 5//2))

    def forward(self, x):
        layer1_out = self.weight_layer_1(x) # weight layer1 output
        layer1_out = F.relu(layer1_out) # weight layer1 output after ac function
        layer2_out = self.weight_layer_2(layer1_out)
        # if size(x) != )? size(F(x)->
        # x-> Linear()-> x_linear = size(F(x)
        final_out = layer2_out + x
        final_out = F.relu(final_out)
        return final_out


if __name__ == '__main__':
    x = torch.randn(size=(1, 3, 10, 10))

    RB = ResidualBlock(in_feature=3)
    final_output = RB(x)
    print("final ouptput", final_output.size())
