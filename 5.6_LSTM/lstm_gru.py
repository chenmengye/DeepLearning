import torch
from torch import nn
import time

# new lstm model
model_lstm = nn.LSTM(
    input_size=4,
    hidden_size=256,
    num_layers=1,
    bidirectional=False
)
print("lstm model param number", sum([p.numel() for p in model_lstm.parameters() if p.requires_grad]))

# new gru model
model_gru = nn.GRU(
    input_size=4,
    hidden_size=256,
    num_layers=1,
    bidirectional=False
)
print("gru model param number", sum([p.numel() for p in model_gru.parameters() if p.requires_grad]))

x = torch.randn(size=(365, 8, 4)) # input data shape: [L, N, H_{in}] = [365, 8, 4]

# lstm inference time consuming and output size
start_time = time.time()
output, (hn, cn) = model_lstm(x)
end_time = time.time()
time_consuming = round((end_time - start_time)*1000) # time consuming in ms
print("lstm time consuming: %d ms" % time_consuming)
print("lsmt ouput shape: ", output.size(), hn.size(), cn.size())

# gru inference time consuming and output size
start_time = time.time()
output, hn = model_gru(x)
end_time = time.time()
time_consuming = round((end_time - start_time)*1000) # time consuming in ms
print("gru time consuming: %d ms" % time_consuming)
print("gru ouput shape: ", output.size(), hn.size())









