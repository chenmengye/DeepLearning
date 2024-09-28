import torch
from torch.utils.data import DataLoader
from config import HP
from utils import load_meta, load_mel
from torch.nn.utils.rnn import pad_sequence


class KWSDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path):
        self.dataset = load_meta(metadata_path)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        cls_id, path = int(item[0]), item[1]
        mel = load_mel(path) # [data_point_dim, sequence_len] = [40, ?]
        # [x,x,x,x,x,0,0]
        # [x,x,x,x,x,x,x]
        return mel.to(HP.device), cls_id # cls_int

    def __len__(self):
        return len(self.dataset)


# batch : 8
def collate_fn(batch):
    # [(mel cls_id),(mel cls_id),(mel cls_id),(mel cls_id)...]
    sorted_batch = sorted(batch, key=lambda b: b[0].size(1), reverse=True)
    # get all mel and pad them: mel defaul dim: [40, ?]=[datapoint_dim, L] -> [L, datapoint_dim]
    mel_list = [item[0].transpose(0, 1) for item in sorted_batch]
    # [sequence, batch, datapoint_dim], [batch, sequence, datapoint_dim]
    mel_padded = pad_sequence(mel_list, batch_first=True)
    labels = torch.LongTensor([item[1] for item in sorted_batch]) # transfer labels to long tensor
    mel_lengths = torch.LongTensor([item.size(0) for item in mel_list])
    return mel_padded, mel_lengths, labels


if __name__ == '__main__':
    torch.manual_seed(1810)
    kws_dataset = KWSDataset(HP.metadata_test_path)
    kws_dataloder = DataLoader(kws_dataset, batch_size=30, shuffle=True, collate_fn=collate_fn)
    for it in kws_dataloder:
        mel_padded, mel_lens, clsid = it
        print(mel_padded.size())
        print(mel_lens)
        break

