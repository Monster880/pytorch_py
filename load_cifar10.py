import glob

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np

label_name = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

label_dict = {}

for idx, name in enumerate(label_name):
    label_dict[name] = idx

def default_loader(path):
    return Image.open(path).convert("RGB")

# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop((28 , 28)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(90),
#     transforms.RandomGrayscale(0.1),
#     transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
#     transforms.ToTensor()
# ])

train_transform = transforms.Compose([
    transforms.RandomCrop(28),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((28 , 28)),
    transforms.ToTensor()
])

class MyDataSet(Dataset):
    def __init__(self, im_list, transform=None, loader = default_loader):
        super(MyDataSet, self).__init__()
        imgs = []

        for im_item in im_list:
            im_label_name = im_item.split("/")[-2]
            imgs.append([im_item, label_dict[im_label_name]])

        self.imgs = imgs
        self.transfrom = transform
        self.loader = loader


    def __getitem__(self, index):
        im_path, im_label = self.imgs[index]
        im_data = self.loader(im_path)

        if self.transfrom is not None:
            im_data = self.transfrom(im_data)

        return im_data,im_label

    def __len__(self):
        return len(self.imgs)

im_train_list = glob.glob("/Users/liding/Documents/pytorch_py/train/*/*.png")
im_test_list = glob.glob("/Users/liding/Documents/pytorch_py/test/*/*.png")

train_dataset = MyDataSet(im_train_list, transform= train_transform)
test_dataset = MyDataSet(im_test_list, transform= transforms.ToTensor())

train_data_loader = DataLoader(dataset = train_dataset, batch_size=6, shuffle=True, num_workers=4)
test_data_loader = DataLoader(dataset = test_dataset, batch_size=6, shuffle=False, num_workers=4)

print("num_of_train", len(train_dataset))
print("num_of_test", len(test_dataset))