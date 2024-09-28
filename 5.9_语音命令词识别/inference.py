import torch
from torch.utils.data import DataLoader
from dataset_kws import KWSDataset, collate_fn
from model import SpeechCommandModel
from config import HP

model = SpeechCommandModel()
# tenor -> device='cuda'/'cpu'
checkpoint = torch.load('G:/05-3x/model_save/model_22_2600.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# test set
testset = KWSDataset(HP.metadata_test_path)
test_loader = DataLoader(testset, batch_size=HP.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn)

model.eval() # set model in test mode (evaluation mode)
# disable dropout
# Norm: GN IN BN, affine -> disable learnable param


total_cnt = 0
correct_cnt = 0

with torch.no_grad():
    for batch in test_loader:
        x, x_lens, y = batch
        pred = model(x, x_lens)
        print("Actual:", y.data.cpu().numpy().tolist()) # actual class label
        print("  Pred:", torch.argmax(pred, dim=-1).data.cpu().numpy().tolist()) # prediction label
        total_cnt += pred.size(0)
        correct_cnt += (torch.argmax(pred, 1) == y).sum()

print('Acc: %.3f' % (correct_cnt/total_cnt))
