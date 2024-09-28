import os
from PIL import Image


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


def load_image(image_path):
    return Image.open(image_path)



