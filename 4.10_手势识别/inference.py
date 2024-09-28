import torch
from torch.utils.data import DataLoader
from dataset_hg import HandGestureDataset
from model import MoocTrialNet
from config import HP

model = MoocTrialNet()
# tenor -> device='cuda'/'cpu'
checkpoint = torch.load('./model_save/model_25_7000.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# test set
# dev datalader(evaluation)
testset = HandGestureDataset(HP.metadata_test_path)
test_loader = DataLoader(testset, batch_size=HP.batch_size, shuffle=True, drop_last=False)

model.eval()

total_cnt = 0
correct_cnt = 0

with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        pred = model(x)
        print(pred)
        total_cnt += pred.size(0)
        correct_cnt += (torch.argmax(pred, 1) == y).sum()

print('Acc: %.3f' % (correct_cnt/total_cnt))
