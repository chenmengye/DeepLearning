from torch import nn
from PIL import Image
from torchvision import transforms # augumentation

transform = transforms.Compose(
    [transforms.ToTensor(), # [0, 1], & to float32
     ]
)

image = Image.open('./lena.jpg')
x = transform(image)    # [0, 1] tensor
x = x.unsqueeze(0) # expand batch dim: [N, C, H, W]=NCHW
batch_size, n_channels, height, width = x.size()
print('x size: ', x.size())

N_OUT_CHS = 32 # 32 kernel / 32 filters
KERNEL_SIZE = 11

conv2d_nn = nn.Conv2d( # Conv1d, Conv3d
    in_channels=n_channels, # 3 for rgb
    out_channels=N_OUT_CHS,
    kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
    stride=1,
    padding=(KERNEL_SIZE//2, KERNEL_SIZE//2),
)


# max pooling: 主要用在关注纹理特征的数据上
pooling_layer = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
x_conv2d_out = conv2d_nn(x) # convolution output
print('conv2d: ', x_conv2d_out.size())
pool_out = pooling_layer(x_conv2d_out)
print('pooling: ', pool_out.size())


# # average pooling: 主要可以平滑图片(特征)
# pooling_layer = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
# x_conv2d_out = conv2d_nn(x) # convolution output
# print('conv2d: ', x_conv2d_out.size())
# pool_out = pooling_layer(x_conv2d_out)
# print('pooling: ', pool_out.size())

# # adaptive pooling: 手动指定size
# pooling_layer = nn.AdaptiveAvgPool2d(output_size=(45, 45))
# x_conv2d_out = conv2d_nn(x) # convolution output
# print('conv2d: ', x_conv2d_out.size())
# pool_out = pooling_layer(x_conv2d_out)
# print('pooling: ', pool_out.size())

