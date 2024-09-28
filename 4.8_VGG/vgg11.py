import torch
from torch import nn


class VGG11(nn.Module):
    def __init__(self, in_channels):
        super(VGG11, self).__init__()
        # input [N, 3, 224, 224]
        self.conv2d_layers = nn.Sequential(  # x=layer(x)

            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), padding=(3//2, 3//2)), # [N, 64, 224, 224]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # [N, 64, 112, 112]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(3 // 2, 3 // 2)), # [N, 128, 112, 112]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # [N, 128, 56, 56]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(3//2, 3//2)), # [N, 256, 56, 56]
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(3 // 2, 3 // 2)),  # [N, 256, 56, 56]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # [N, 256, 28, 28]
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(3//2, 3//2)), #[N, 512, 28, 28]
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(3 // 2, 3 // 2)), # [N, 512, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # [N, 512, 14, 14]
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(3//2, 3//2)), #[N, 512, 14, 14]
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(3 // 2, 3 // 2)),  # [N, 512, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # [N, 512, 7, 7]
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096), # [N, 512*7*7] -> [N, 4096]
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096), # [N, 4096]
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000),
        )

    def forward(self, x):
        out_from_conv2d = self.conv2d_layers(x) # [N, 512, 7, 7]
        out_from_conv2d_flatten = out_from_conv2d.view(out_from_conv2d.size(0), -1) # [N, 512*7*7]
        final = self.fc_layers(out_from_conv2d_flatten)
        return final


if __name__ == '__main__':
    # input: [8, 3, 224, 224]
    x = torch.randn(size=(8, 3, 224, 224))
    vgg11 = VGG11(in_channels=3)
    output = vgg11(x)
    print('output shape: ', output.size()) #[N, 1000]

