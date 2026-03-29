import torch
import os
import os
import PIL
from PIL import Image
import pandas as pd
from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class BSVDataset1m(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        df = pd.read_csv(root)

        self.imgs = df["new_path"].to_list()


    def __getitem__(self, index):
        path = '/home/huangyingjing/data/baidu_pano/'+self.imgs[index]
        name = os.path.basename(path).split(".")[0]

        # 读取图像文件
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

            sample = (img, name)
            return sample

    def __len__(self):
        return len(self.imgs)

class BSVDataset1mtest(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        df = pd.read_csv(root)

        self.imgs = df["new_path"].to_list()


    def __getitem__(self, index):
        path = '/home/huangyingjing/data/baidu_pano/'+self.imgs[index]
        name = os.path.basename(path)
        # 读取图像文件
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

            sample = (img, name)
            return sample

    def __len__(self):
        return len(self.imgs)


class GSVDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # self.imgs = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root) for f in filenames if f.endswith('.jpg')]
        df = pd.read_csv(root)

        self.imgs = df["image_path"].to_list()


    def __getitem__(self, index):
        path = self.imgs[index]
        name = os.path.basename(path).split(".")[0]

        # 读取图像文件
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

            sample = (img, name)
            return sample

    def __len__(self):
        return len(self.imgs)
    
class GSVDataset1m(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        df = pd.read_pickle(root)

        self.imgs = df["path"].to_list()


    def __getitem__(self, index):
        path = self.imgs[index]
        name = os.path.basename(path).split(".")[0]

        # 读取图像文件
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

            sample = (img, path)
            return sample

    def __len__(self):
        return len(self.imgs)



class PlaceDataset(torch.utils.data.Dataset):
    def __init__(self, path, is_train=True, args=None):
        # 读取数据
        self.df = pd.read_pickle(path)
        
        # 筛选出 'set' 列为 'train' 的行or val
        if is_train:
            self.df = self.df[self.df['set'] == 'train']
            
        else:
            self.df = self.df[self.df['set'] == 'val']

        # 获取图像路径和标签
        self.img_paths = self.df['path'].values
        self.labels = self.df['label_id'].values
        
        # 预处理变换
        self.transform = build_transform(is_train, args)

    def __len__(self):
        # 返回数据集大小
        return len(self.img_paths)

    def __getitem__(self, index):
        # 获取图像路径和标签
        img_path = self.img_paths[index]
        label = self.labels[index]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 预处理图像
        if self.transform:
            image = self.transform(image)
        
        return (image, label)


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
