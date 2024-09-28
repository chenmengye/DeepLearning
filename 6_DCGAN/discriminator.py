# Discriminator : Binary classification model
import torch
from torch import nn
from config import HP


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential( # 1. shape transform 2. use conv layer as "feature extraction"
            # conv layer : 1
            nn.Conv2d(in_channels=HP.data_channels, # [N, 16, 32, 32]
                      out_channels=16,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1),
                      bias=False),
            nn.LeakyReLU(0.2),
            # conv layer : 2
            nn.Conv2d(in_channels=16,  # [N, 32, 16, 16]
                      out_channels=32,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # conv layer : 3
            nn.Conv2d(in_channels=32,  # [N, 64, 8, 8]
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # conv layer : 4
            nn.Conv2d(in_channels=64,  # [N, 128, 4, 4]
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # conv layer : 5
            nn.Conv2d(in_channels=128,  # [N, 256, 2, 2]
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(2, 2),
                      padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        self.linear = nn.Linear(256*2*2, 1)
        self.out_ac = nn.Sigmoid()

    def forward(self, image):
        out_d = self.discriminator(image) # image [N, 3, 64, 64] -> [N, 256, 2, 2]
        out_d = out_d.view(-1, 256*2*2) # tensor flatten
        return self.out_ac(self.linear(out_d))

    @staticmethod
    def weights_init(layer):
        layer_class_name = layer.__class__.__name__
        if 'Conv' in layer_class_name:
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
        elif 'BatchNorm' in layer_class_name:
            nn.init.normal_(layer.weight.data, 1.0, 0.02)
            nn.init.normal_(layer.bias.data, 0.)


if __name__ == '__main__':
    g_z = torch.randn(size=(64, 3, 64, 64))
    D = Discriminator()
    d_out = D(g_z)
    print(d_out.size())



