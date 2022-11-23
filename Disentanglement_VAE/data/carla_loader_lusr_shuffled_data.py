''' Data를 Weather 별로 list에 담아서 return '''

#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Thu Nov 22 12:09:27 2018
Info:
'''

import glob

import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from imgaug import augmenters as iaa
from utils.helper import RandomTransWrapper
import random

class CarlaH5Data_Simple():
    def __init__(self,
                 train_pair_folders,
                 eval_pair_folders,
                 batch_size=4, num_workers=4, distributed=False):

        self.loaders = {
            "train": torch.utils.data.DataLoader(
                CarlaH5Dataset_Simple(
                    data_pair_dir=train_pair_folders,
                    train_eval_flag="train"),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=True
            ),
            "eval": torch.utils.data.DataLoader(
                CarlaH5Dataset_Simple(
                    data_pair_dir=eval_pair_folders,
                    train_eval_flag="eval"),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=False
            )}


class CarlaH5Dataset_Simple(Dataset):
    def __init__(self, data_pair_dir,
                 train_eval_flag="train", sequence_len=200):

        self.data_pair_dir = data_pair_dir

        self.data_pair_list = []
        num_pair_data = len(self.data_pair_dir)
        for i in range(num_pair_data):
            self.data_pair_list.append(glob.glob(self.data_pair_dir[i]+'*.h5'))

        self.sequnece_len = sequence_len
        self.train_eval_flag = train_eval_flag

        self.build_transform()

    def build_transform(self):
        if self.train_eval_flag == "train":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])

    def __len__(self):
        # return self.sequnece_len * len(self.data_pair_list[0])
        return len(self.data_pair_list[0])

    def __getitem__(self, idx):
        data_idx = idx
        file_name1 = self.data_pair_list[0][data_idx]
        tmp_batch_size = 16

        if self.train_eval_flag == "train":
            rand_range = random.randrange(200 - tmp_batch_size)
        elif self.train_eval_flag == "eval":
            rand_range = 0

        with h5py.File(file_name1, 'r') as h5_file:
            img_batch1 = np.array(h5_file['rgb'])[rand_range:rand_range+tmp_batch_size]
            img_batch1 = torch.cat([self.transform(out).unsqueeze(0) for out in img_batch1], dim=0)

        file_name2 = self.data_pair_list[1][data_idx]
        with h5py.File(file_name2, 'r') as h5_file:
            img_batch2 = np.array(h5_file['rgb'])[rand_range:rand_range+tmp_batch_size]
            img_batch2 = torch.cat([self.transform(out).unsqueeze(0) for out in img_batch2], dim=0)

        file_name3 = self.data_pair_list[2][data_idx]
        with h5py.File(file_name3, 'r') as h5_file:
            img_batch3 = np.array(h5_file['rgb'])[rand_range:rand_range+tmp_batch_size]
            img_batch3 = torch.cat([self.transform(out).unsqueeze(0) for out in img_batch3], dim=0)

        file_name4 = self.data_pair_list[3][data_idx]
        with h5py.File(file_name4, 'r') as h5_file:
            img_batch4 = np.array(h5_file['rgb'])[rand_range:rand_range+tmp_batch_size]
            img_batch4 = torch.cat([self.transform(out).unsqueeze(0) for out in img_batch4], dim=0)

            # add preprocess
        return img_batch1, img_batch2, img_batch3, img_batch4

def main():
    train_pair = ["/SSD1/datasets/carla/pair_db/gen_data_W1/", "/SSD1/datasets/carla/pair_db/gen_data_W3/",
                      "/SSD1/datasets/carla/pair_db/gen_data_W6/", "/SSD1/datasets/carla/pair_db/gen_data_W8/"]
    eval_pair = ["/SSD1/datasets/carla/pair_db/gen_data_W1/", "/SSD1/datasets/carla/pair_db/gen_data_W3/",
                      "/SSD1/datasets/carla/pair_db/gen_data_W6/", "/SSD1/datasets/carla/pair_db/gen_data_W8/"]
    batch_size = 1
    workers = 1

    carla_data = CarlaH5Data_Simple(
        train_pair_folders=train_pair,
        eval_pair_folders=eval_pair,
        batch_size=batch_size
    )

    train_loader = carla_data.loaders["train"]
    eval_loader = carla_data.loaders["eval"]

    for i, (img1, img2, img3, img4) in enumerate(train_loader):
        print(i)

if __name__ == '__main__':
    main()