import torch
from torch import nn

# new a RNN model
model_rnn = nn.RNN(
    input_size=4,       # input data channel
    hidden_size=128,    # rnn hidden dim, 62-1024: 128/256/512
    bias=True,
    num_layers=1,       # rnn layer number
    bidirectional=False #
)

x = torch.randn(size=(31, 8, 4))
# `(L, N, H_{in})`
# L: sequnce length: L=31
# N: batch size
# H_{in}: input channle, K line data input channel is 4
output, hn = model_rnn(x) # hn: hidden state
print("output size", output.size(), "hn size", hn.size())
# output size torch.Size([31, 8, 128]) hn size torch.Size([1, 8, 128])

