import os
import torchaudio


# 获取某个文件夹下面所有后缀为suffix的文件，返回path的list
def recursive_fetching(root, suffix=['jpg', 'png']):
    all_file_path = []

    def get_all_files(path):
        all_file_list = os.listdir(path)
        # 遍历该文件夹下的所有目录或者文件
        for file in all_file_list:
            filepath = os.path.join(path, file)
            # 如果是文件夹，递归调用函数
            if os.path.isdir(filepath):
                get_all_files(filepath)
            # 如果不是文件夹，保存文件路径及文件名
            elif os.path.isfile(filepath):
                all_file_path.append(filepath)

    get_all_files(root)

    file_paths = [it for it in all_file_path if os.path.split(it)[-1].split('.')[-1].lower() in suffix]

    return file_paths


def load_meta(meta_path):
    with open(meta_path, 'r') as fr:
        return [line.strip().split('|') for line in fr.readlines()]


# 加载数据，并获取转换后的音频的mel谱特征：
def load_mel(audio_path):
    wave, sampling_rate = torchaudio.load(audio_path)
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate, n_mels=40)(wave).squeeze(0)
    return mel


if __name__ == '__main__':
    mel_ = load_mel('./data/go/20174140_nohash_0.wav')
    print("mel_", mel_.size())
    data_point_dim = mel_.size(0)
    sequence_data_length = mel_.size(1)
    print("data_point_dim", data_point_dim)
    print("sequence_data_length", sequence_data_length)

