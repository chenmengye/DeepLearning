from torch import nn

model = nn.Linear(in_features=16, out_features=128)
# Xavier
print(model.weight)
nn.init.xavier_uniform_(model.weight, gain=nn.init.calculate_gain('tanh'))
print(model.weight)
# Kaiming
nn.init.kaiming_uniform_(model.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
print(model.weight)



