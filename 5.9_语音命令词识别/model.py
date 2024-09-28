import torch
from torch import nn
from torch.nn import functional as F
from config import HP
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def mish(x): # [N, ....]
    return x*torch.tanh(F.softplus(x))


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return mish(x)


class SpeechCommandModel(nn.Module):
    def __init__(self):
        super(SpeechCommandModel, self).__init__()

        self.rnn = nn.GRU(
            input_size=HP.data_point_channel, # 数据点的channel=mel 点数 = 40
            hidden_size=HP.rnn_hidden_dim, # rnn hidden layer dimension
            num_layers=HP.rnn_layer_num, # two layers rnn
            bidirectional=HP.is_bidirection, # True default
        )
        # ** output ** of shape `(seq_len, batch, num_directions * hidden_size)`
        fc_in_dim = 2 * HP.rnn_hidden_dim if HP.is_bidirection else HP.rnn_hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, 1024),
            Mish(),
            nn.Dropout(HP.fc_drop),
            nn.Linear(1024, 512),
            Mish(),
            nn.Dropout(HP.fc_drop),
            nn.Linear(512, HP.classes_num)
        )

    def forward(self, mel_input, mel_lengths):
        # mel_input: [batch, sequence_len, datapoint_dim], mel_lengths: [81,75, 45, ...]
        mel_input = mel_input.permute(1, 0, 2) # mel_input: [sequence_len, batch, datapoint_dim]
        mel_packed = pack_padded_sequence(mel_input, mel_lengths)
        output_packed, hn = self.rnn(mel_packed)
        output, _ = pad_packed_sequence(output_packed) # [sequence_len, batch, rnn_hidden_dim*(?)]
        if HP.is_bidirection:
            forward_feature = output[-1, :, :HP.rnn_hidden_dim] # [batch, rnn_hidden]
            backward_feature = output[0, :, HP.rnn_hidden_dim:] # [batch, rnn_hidden]
            fc_in = torch.cat((forward_feature, backward_feature), dim=-1)
            cls_output = self.fc(fc_in)
        else:
            cls_output = self.fc(output[-1, :])

        return cls_output


if __name__ == '__main__':
    from dataset_kws import KWSDataset, collate_fn
    from torch.utils.data import DataLoader
    torch.manual_seed(1810)
    kws_dataset = KWSDataset(HP.metadata_test_path)
    kws_dataloder = DataLoader(kws_dataset, batch_size=30, shuffle=True, collate_fn=collate_fn)
    kws = SpeechCommandModel()
    for it in kws_dataloder:
        mel_padded, mel_lens, clsid = it
        out = kws(mel_padded, mel_lens)
        print(out.size())
        break













