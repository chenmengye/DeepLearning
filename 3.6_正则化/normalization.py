import torch
from torch import nn
import torch.nn.functional as F

output_from_pre_layer = torch.randn(size=(8, 224, 224, 16)) # RGB image: [224,224,3], NHWC

""" NHWC
N: batch size : 8
H: height: 224,
W: width: 224,
C: channel: 16

NCHW, NHWC
"""

""" batch norm
在N 求E(x), Var(x)
适用于大的 batch size
不太适用于变长数据：text, speech
"""
bn_norm = nn.BatchNorm2d(num_features=16) # input shape: NCHW
norm_out = bn_norm(output_from_pre_layer.permute(0, 3, 1, 2)) # NHWC -> NCHW
print('norm from bn', norm_out.shape)

""" layer normalization
在[224,224,16]求E(x), Var(x)
对batch size 不敏感， 适用sequence data(序列变长的数据): RNN/Transformer
"""
ln_norm = nn.LayerNorm([224, 224, 16]) # input shape: [N, *]
norm_out = ln_norm(output_from_pre_layer)
print('norm from ln', norm_out.shape)

""" instance normalization
在channel 这个维度上求的E(x), Var(x)
适用GAN(生成式神经网络)
"""
in_norm = nn.InstanceNorm2d(16) # input shape: NCHW
norm_out = in_norm(output_from_pre_layer.permute(0, 3, 1, 2))
print('norm from in', norm_out.shape)

""" group normalization
在分组后的group上求的E(x), Var(x): [224, 224, group_number， 16/group_number]
group number: 需要精心设置
"""
gn_norm = nn.GroupNorm(num_groups=4, num_channels=16) # input shape: (N, C, *)
norm_out = gn_norm(output_from_pre_layer.permute(0, 3, 1, 2))
print('norm from gn', norm_out.shape)

final_output = F.relu(norm_out)
