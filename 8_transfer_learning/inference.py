import torch
from torch.utils.data import DataLoader
from dataset_action import ActionDataset
from model import MoocTLNet
from config import HP

# change to cpu
HP.device = 'cpu'
# new a model instance
model = MoocTLNet()
# tenor -> device='cuda'/'cpu'
checkpoint = torch.load('./model_save/model_61_3000.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# test set
testset = ActionDataset(HP.metadata_test_path)
test_loader = DataLoader(testset, batch_size=HP.batch_size, shuffle=True, drop_last=False)
model.eval()

total_cnt = 0
correct_cnt = 0

import time
start_st = time.time()
with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        pred = model(x)
        # print(pred)
        total_cnt += pred.size(0)
        correct_cnt += (torch.argmax(pred, 1) == y).sum()
print(time.time()-start_st)
print('Acc: %.3f' % (correct_cnt/total_cnt))
