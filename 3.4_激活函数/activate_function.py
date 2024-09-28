import torch
from torch import nn
import torch.nn.functional as F

# define a layer
layer = nn.Linear(in_features=16, out_features=5)
x = torch.randn(size=(8, 16))
layer_output = layer(x)
print(layer_output.size())

# sigmoid
layer_output = F.sigmoid(layer_output)
print(layer_output.size())

# relu: ****
layer_output = F.relu(layer_output)
print(layer_output.size())

# leaky_relu
layer_output = F.leaky_relu(layer_output)
print(layer_output.size())


def mish(x):
    return x*F.tanh(F.softplus(x))


# mish: ****
layer_output = mish(layer_output)
print(layer_output.size())
