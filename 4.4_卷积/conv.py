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
x_out = conv2d_nn(x)
print('x_out size: ', x_out.size())


