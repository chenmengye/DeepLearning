import torchvision.transforms as T
from PIL import Image

trans = T.Compose(
    [
        T.ToTensor(), # [0, 1] & float32
        T.RandomRotation(45),
        T.RandomAffine(45),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)
image = Image.open('./lena.jpg')
print(image)
t_out_image = trans(image)
print(t_out_image.size())
image.show()
input()
T.RandomRotation(90)(image).show()
