import torch
from torch import nn


# param number calc
def model_param_number_calc(model_):
    return sum([p.numel() for p in model_.parameters() if p.requires_grad])

# input: [10, 10, 3] -> output: [10, 10, 30]


# fully-connected nn
model_fc = nn.Linear(in_features=10*10*3, out_features=10*10*30, bias=True)
print('fc', model_param_number_calc(model_fc))

# basic conv2d
model_basic_conv2d = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=(10, 10), bias=True)
print('basic_conv2d', model_param_number_calc(model_basic_conv2d))

# dilated conv2d
model_dilated_conv2d = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=(10, 10), bias=True, dilation=(2, 2))
print('model_dilated_conv2d', model_param_number_calc(model_dilated_conv2d))

# group conv2d
model_group_conv2d = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=(10, 10), bias=True, groups=3)
print('model_group_conv2d', model_param_number_calc(model_group_conv2d))

# point-wise conv2d
model_pointwise_conv2d = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=(1, 1), bias=True)
print('model_pointwise_conv2d', model_param_number_calc(model_pointwise_conv2d))

# deep separable conv2d
depth_conv2d = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(10, 10), groups=3)
point_conv2d = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=(1, 1))
print('model_ds_conv2d', model_param_number_calc(depth_conv2d)+model_param_number_calc(point_conv2d))

# transpose conv2d
transpose_conv2d = nn.ConvTranspose2d(in_channels=3, out_channels=30, kernel_size=(10, 10))
print(transpose_conv2d(torch.randn(size=(1, 3, 10, 10))).size()) # NCHW


