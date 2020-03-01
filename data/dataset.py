import os
import random
import torchvision.transforms as T
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader

from config import opt




class DogCat(Dataset):
    def __init__(self, root, transforms=None, mode='train'):
        self.mode = mode
        if self.mode == "train" or self.mode == "eval":
            root += "\\train"
        elif self.mode == "test":
            root += "\\test"
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        img_nums = len(imgs)
        if self.mode != "test":
            random.shuffle(imgs)

            if self.mode == "train":
                self.imgs = imgs[:int(0.7*img_nums)]
            elif self.mode == "eval":
                self.imgs = imgs[int(0.99*img_nums):]
        else:
            self.imgs = sorted(imgs, key=lambda x: int(x.split('\\')[-1].split('.')[-2]))
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std= [0.229, 0.224, 0.225])
            if self.mode == "train":
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomCrop(224),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(244),
                    T.CenterCrop(224),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        if self.mode == "test":
            label = img_path.split('\\')[-1].split('.')[-2]
        else:
            label = 1 if 'dog' in img_path else 0

        data = Image.open(img_path)
        try:
            data = self.transforms(data)
        except IOError:
            print(img_path)
            print(np.array(data))
        return data, label

    def __len__(self):
        return len(self.imgs)


train_loader = DataLoader(DogCat(root=opt.root, mode="train"), batch_size=opt.batch_size, num_workers=4)
eval_loader = DataLoader(DogCat(root=opt.root, mode="eval"), batch_size=opt.batch_size, num_workers=4)
test_loader = DataLoader(DogCat(root=opt.root, mode="test"), batch_size=opt.batch_size, num_workers=4)