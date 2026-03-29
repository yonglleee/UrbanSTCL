import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
# from datasets import BSVDataset1mtest,GSVDataset1m
from PIL import Image
# from util.pos_embed import interpolate_pos_embed
# import models_vit
# from datasets import build_transform

import torchvision.models as torchvision_models

import vits


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
class Perception(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        df = pd.read_pickle(root)

        self.imgs = df["path"].to_list()
        self.label = df["label"].to_list()


    def __getitem__(self, index):
        path = self.imgs[index]
        name = os.path.basename(path).split(".")[0]
        label = self.label[index]

        # 读取图像文件
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

            sample = (img, path,label)
            return sample

    def __len__(self):
        return len(self.imgs)
# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)


def prepare_model(chkpt_dir, arch='vit_base'): 
    # create model
    args.arch = arch
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('vit'):
        model = vits.__dict__[args.arch]()
        linear_keyword = 'head'
    else:
        model = torchvision_models.__dict__[args.arch]()
        linear_keyword = 'fc'


    print("=> loading checkpoint '{}'".format(chkpt_dir))
    checkpoint = torch.load(chkpt_dir, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    args.start_epoch = 0
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

    print("=> loaded pre-trained model '{}'".format(chkpt_dir))

    return model


def main(args):
    # 设置预处理转换
    # simple augmentation
    args.input_size = 224
    transform_test = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=3),
            # transforms.RandomResizedCrop(args.input_size, scale=(1.0), interpolation=3),  # 3 is bicubic
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.58766386,0.62574633, 0.66660842], std=[0.09801628, 0.09911405, 0.12225879]) 
            ])
    
    # 加载模型
        # 加载模型并移动到指定 GPU
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = prepare_model(args.model_path).to(device)
    # print(model)
    model.eval()

    dataset = Perception(args.data_path, transform=transform_test)

    # 用于存储特征和文件路径的列表
    features_data = []
    # 使用 DataLoader 加载数据集
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,num_workers=32)

    # 对每个批次的数据进行迭代
    for batch in tqdm(dataloader):
        (imgs, files,labels) = batch
        # 将数据移动到指定 GPU
        imgs = imgs.to(device)
        # print(imgs.shape)

        # 使用模型提取特征
        with torch.no_grad():
            cls_tokens = model(imgs)
            # print(cls_tokens.shape)

            cls_tokens = cls_tokens.cpu().numpy()
            # print(cls_tokens)
            # print(loss.shape)
        # 将城市和特征添加到列表

        # labels = labels.tolist()

        for path, label ,feature in zip(files, labels,cls_tokens):
            # print(label)
       
        # 展平特征数组并转换为列表
            feature_flat = feature.flatten().tolist()
            features_data.append({'path': path, 'label': label.item(),**{f'{i}': feature_flat[i] for i in range(len(feature_flat))}})
            
    # 转换为 DataFrame
    df = pd.DataFrame(features_data)
    # 保存特征数据到 CSV 文件
    df.to_pickle(args.output_dir)

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from images using MAE")
    parser.add_argument("--data_path", type=str, default="/home/liyong/svpretrain/process/test_img_merged_path.pkl", help="Root directory containing city images")
    parser.add_argument("--output_dir", type=str, default="/home/liyong/svpretrain/output_dir/feature/mocov3_vit_base_200ep_gsv1m.pkl", help="Output directory to save feature CSV files")
    parser.add_argument("--model_path", type=str, default="/home/huangyingjing/sv_learn/models/Mocov3VITB-self-cities10-1m/checkpoint_199.pth.tar", help="")

    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for feature extraction")
    parser.add_argument("--gpu_id", type=int, default=4, help="GPU ID to use")
    args = parser.parse_args()



    print(args)
    main(args)


# python /home/liyong/code/svpretrain/moco-v3/main_test_gsv1m_feature.py \
# --data_path /home/liyong/code/svpretrain/Perception/filtered_labels.p  \
# --model_path /home/liyong/code/svpretrain/output_dir/checkpoint/mocov3_paper/vit-b-300ep.pth.tar \
# --output_dir  /home/liyong/code/svpretrain/Perception/feature/imagenet.pkl \
# --batch_size 128 --gpu_id 1