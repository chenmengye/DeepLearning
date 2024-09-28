import torch
from torch import nn
import time

# multi layer and bi-directional gru
mode_gru = nn.GRU(
    input_size=4,
    hidden_size=256,
    num_layers=3, # normally, 2-4
    bidirectional=True, # enable bi-directional
)

x = torch.randn(size=(365, 8, 4))

start_time = time.time()
output, hn = mode_gru(x)
end_time = time.time()
time_consuming = round((end_time - start_time)*1000)
print("time consuming: %d ms" % time_consuming)
print(output.size(), hn.size())
# **output** of shape `(seq_len, batch, num_directions * hidden_size)`
# **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`

