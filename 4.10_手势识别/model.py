import torch
from torch import nn
from torch.nn import functional as F
from config import HP


def mish(x): # [N, ....]
    return x*torch.tanh(F.softplus(x))


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return mish(x)


class DSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DSConv2d, self).__init__()
        assert kernel_size % 2 == 1, "odd needed!"
        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(kernel_size//2, kernel_size//2),
            groups=in_channels
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1)
        )

    def forward(self, input_x):
        out = self.depth_conv(input_x)
        out_final = self.pointwise_conv(out)
        return out_final


class MoocTrialBlock(nn.Module):
    def __init__(self, in_channels):
        super(MoocTrialBlock, self).__init__()
        self.left_flow = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(in_channels),
            Mish(),
            DSConv2d(in_channels=in_channels,out_channels=in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            Mish(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(7, 7), padding=(7//2, 7//2))
        )
        self.right_flow = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(7, 7), padding=(7 // 2, 7 // 2)),
            nn.BatchNorm2d(in_channels),
            Mish(),
            DSConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3),
            nn.BatchNorm2d(in_channels),
            Mish(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1)),
        )

    def forward(self, input_ft):
        out = self.left_flow(input_ft) + self.right_flow(input_ft) + input_ft
        out_final = mish(out)
        return out_final


class MoocTrialNet(nn.Module):
    def __init__(self):
        super(MoocTrialNet, self).__init__()

        self.mtn_conv = nn.Sequential(
            nn.Conv2d(in_channels=HP.data_channels, out_channels=64, kernel_size=(3, 3), padding=(3//2, 3//2)),
            nn.BatchNorm2d(64),
            Mish(),
            MoocTrialBlock(in_channels=64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(3 // 2, 3 // 2)),
            nn.BatchNorm2d(128),
            Mish(),
            MoocTrialBlock(in_channels=128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(3 // 2, 3 // 2)),
            nn.BatchNorm2d(256),
            Mish(),
            MoocTrialBlock(in_channels=256),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            MoocTrialBlock(in_channels=256),
            MoocTrialBlock(in_channels=256),
            MoocTrialBlock(in_channels=256),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # original input shape: [N, 3, 112,112] -> [N, 256, 7, 7]
        )

        self.mtn_fc = nn.Sequential(
            nn.Linear(in_features=256*7*7, out_features=2048),
            Mish(),
            nn.Dropout(HP.fc_drop_prob),

            nn.Linear(in_features=2048, out_features=1024),
            Mish(),
            nn.Dropout(HP.fc_drop_prob),

            nn.Linear(in_features=1024, out_features=HP.classes_num)
        )

    def forward(self, input_x):
        out = self.mtn_conv(input_x) # [N, 256, 7, 7]
        out_final = self.mtn_fc(out.view(input_x.size(0), -1)) # fc input shape: [N, *, dim], -> [N, 256*7*7]
        return out_final


# if __name__ == '__main__':
#     model = MoocTrialNet()
#     x = torch.randn(size=(5, 3, 112, 112))
#     y_pred = model(x)
#     print(y_pred.size()) # [N, classes_num]












