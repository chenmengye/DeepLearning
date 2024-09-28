import os
from config import HP
from utils import recursive_fetching
import random
import json
from scipy.io import wavfile as wf

random.seed(HP.seed)

"""
# 因为“_background_noise_”这个文件夹下的数据不是一秒钟长，
# 所以要把它剪成一个个长度为1s的音频片段, 然后删除原始文件, 放在bgn文件夹下
def chop_bgn():
    waves_files = recursive_fetching(os.path.join(HP.data_root, "_background_noise_"), suffix=["wav", "WAV"])
    for wave_f in waves_files:
        file_name = os.path.split(wave_f)[-1]
        sampling_rate, data = wf.read(wave_f) # 打开音频文件
        data_len = data.shape[0] # 总共的采样点个数
        len_200ms = int(sampling_rate*200/1000) # 200 ms的采样点个数
        count = round(data_len/len_200ms) # 总共取count个长度为1s的音频

        for i in range(count):
            segment = data[i*len_200ms: i*len_200ms+sampling_rate]
            ouput_file_name = "seg-%d-%s" % (i, file_name)
            wf.write(os.path.join(HP.data_root, 'bgn', ouput_file_name),sampling_rate, segment)


chop_bgn() # 将_background_noise_下面的背景噪声，剪切成1s长度的一个个的片段
"""
# 构建一个类别到id的映射
cls_mapper = {
    "cls2id": {"bgn": 0, "down": 1, "go": 2, "left": 3, "off": 4, "on": 5, "right": 6, "stop": 7},
    "id2cls": {0: "bgn", 1: "down", 2: "go", 3: "left", 4: "off", 5: "on", 6: "right", 7: "stop"}
}
json.dump(cls_mapper, open(HP.cls_mapper_path, 'w')) # 保存成json

# 获取train和test数据集，并将他们合并
dataset = recursive_fetching(HP.data_root, suffix=['wav'])
dataset_num = len(dataset)
print("Total Items: %d" % dataset_num)
random.shuffle(dataset) # 随机打乱
# # 数据集的每一个类别及对应的数据list
# dataset_dict = {
#     0: [...]
#     1: [...]
#     2: ["./data/go/0a9f9af7_nohash_0.wav", "./data/go/0a9f9af7_nohash_1.wav", ...]
#     ...
#     7: [...]
# }

dataset_dict = {}
for it in dataset:
    cls_name = os.path.split(os.path.split(it)[0])[-1] # ".../data/bgn/seg-11-xxxx.wav"
    cls_id = cls_mapper["cls2id"][cls_name]
    if cls_id not in dataset_dict:
        dataset_dict[cls_id] = [it]
    else:
        dataset_dict[cls_id].append(it)

# 每个类别按照比例分到train/eval/test
train_ratio, eval_ratio, test_ratio = 0.8, 0.1, 0.1
train_set, eval_set, test_set = [], [], []
for _, set_list in dataset_dict.items():
    length = len(set_list)
    train_num, eval_num = int(length*train_ratio), int(length*eval_ratio)
    test_num = length - train_num - eval_num
    random.shuffle(set_list)
    train_set.extend(set_list[:train_num])
    eval_set.extend(set_list[train_num:train_num+eval_num])
    test_set.extend(set_list[train_num+eval_num:])

# 再次随机打乱
random.shuffle(train_set)
random.shuffle(eval_set)
random.shuffle(test_set)
print("train set : eval set : test set -> ", len(train_set), len(eval_set), len(test_set))

# 写入meta file
with open(HP.metadata_train_path, 'w') as fw:
    for path in train_set:
        cls_name = os.path.split(os.path.split(path)[0])[-1]  # ".../data/bgn/seg-11-xxxx.wav"
        cls_id = cls_mapper["cls2id"][cls_name]
        fw.write("%d|%s\n" % (cls_id, path))

with open( HP.metadata_eval_path, 'w') as fw:
    for path in eval_set:
        cls_name = os.path.split(os.path.split(path)[0])[-1]  # ".../data/bgn/seg-11-xxxx.wav"
        cls_id = cls_mapper["cls2id"][cls_name]
        fw.write("%d|%s\n" % (cls_id, path))

with open(HP.metadata_test_path, 'w') as fw:
    for path in test_set:
        cls_name = os.path.split(os.path.split(path)[0])[-1]  # ".../data/bgn/seg-11-xxxx.wav"
        cls_id = cls_mapper["cls2id"][cls_name]
        fw.write("%d|%s\n" % (cls_id, path))

