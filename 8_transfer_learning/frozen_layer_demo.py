import torch
from torch import nn

net = nn.Linear(10, 100)
for p in net.parameters():
    p.requires_grad = False

for k, v in net.named_parameters():
    if v.requires_grad:
        print(k, v)

