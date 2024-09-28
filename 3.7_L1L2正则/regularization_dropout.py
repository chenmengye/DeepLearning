import torch
from torch import nn
from torch.nn import functional as F



# fake data
n_item = 1000
n_feature = 2

torch.manual_seed(123)
data_x = torch.randn(size=(n_item, n_feature)).float()
data_y = torch.where(torch.subtract(data_x[:, 0]*0.5, data_x[:, 1]*1.5)+0.02 > 0, 0, 1).long()

data_y = F.one_hot(data_y)  # one hot encode
""" shape: [n_item, 2]
tensor([[1, 0],
        [1, 0],
        [0, 1],
        ...,
        [1, 0],
        [1, 0],
        [1, 0]])
"""


class BinaryClassificationModel(nn.Module):
    def __init__(self, in_feature):
        super(BinaryClassificationModel, self).__init__()
        """ single perceptron 
        self.layer_1 = nn.Linear(in_features=in_feature, out_features=2, bias=True)
        """

        """ multi perceptron """
        self.layer_1 = nn.Linear(in_features=in_feature, out_features=128, bias=True)
        self.layer_2 = nn.Linear(in_features=128, out_features=512, bias=True)
        # 。。。
        self.layer_final = nn.Linear(in_features=512, out_features=2, bias=True)

        self.drop_layer1 = nn.Dropout(0.1)
        self.drop_layer2 = nn.Dropout(0.3)

    def forward(self, x):
        layer_1_output = F.sigmoid(self.layer_1(x))
        layer_1_output = self.drop_layer1(layer_1_output)
        layer_2_output = F.sigmoid(self.layer_2(layer_1_output))
        layer_2_output = self.drop_layer2(layer_2_output)
        output = F.sigmoid(self.layer_final(layer_2_output))
        return output


# hyper parameters
learning_rate = 0.01
epochs = 100

model = BinaryClassificationModel(n_feature)

opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

criteria = nn.BCELoss()


# train loop
for epoch in range(1000):
    for step in range(n_item):
        x = data_x[step]
        y = data_y[step]

        opt.zero_grad()

        y_hat = model(x.unsqueeze(0)) # [1, 2]

        layer_1_w = torch.cat([x.view(-1) for p in model.layer_1.parameters()])
        layer_2_w = torch.cat([x.view(-1) for p in model.layer_2.parameters()])
        layer_final_w = torch.cat([x.view(-1) for p in model.layer_final.parameters()])
        all_weights = torch.cat([layer_1_w, layer_2_w, layer_final_w], dim=-1)
        l2_norm = torch.norm(all_weights, 2)

        # [1, 2]: [[0.9, 0.1]]
        loss = criteria(y_hat, y.unsqueeze(0).float()) + l2_norm

        loss.backward()
        opt.step()
    print('Epoch: %03d, Loss: %.3f' % (epoch, loss.item()))





