import torch
from torch.utils.data import DataLoader
from config import HP
from utils import load_meta, load_image
from torchvision import transforms as T

hg_transform = T.Compose([
    T.Resize((112, 112)),                   # 保证同样输入的shape
    # T.RandomRotation(degrees=45),           # 减小倾斜图片影响
    # T.GaussianBlur(kernel_size=(3, 3)),     # 抑制模糊图片的影响
    # T.RandomHorizontalFlip(),               # 左右手
    T.ToTensor(),                           # 归一化 & float32 tensor
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))   # 标准化
])


class HandGestureDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path):
        self.dataset = load_meta(metadata_path) # [(0, image_path), () ,...]

    def __getitem__(self, idx):
        item = self.dataset[idx]
        cls_id, path = int(item[0]), item[1]
        image = load_image(path)
        return hg_transform(image).to(HP.device), cls_id

    def __len__(self):
        return len(self.dataset)
