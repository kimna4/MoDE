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
from helper import RandomTransWrapper
import random

class CarlaH5Data_Simple():
    def __init__(self,
                 train_folder,
                 fine_folder,
                 eval_folder,
                 batch_size=4, num_workers=4):

        self.loaders = {
            "train": torch.utils.data.DataLoader(
                CarlaH5Dataset_Simple(
                    data_dir=train_folder,
                    train_eval_flag="train"),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=True
            ),
            "fine": torch.utils.data.DataLoader(
                CarlaH5Dataset_Simple(
                    data_dir=fine_folder,
                    train_eval_flag="fine"),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=True
            ),
            "eval": torch.utils.data.DataLoader(
                CarlaH5Dataset_Simple(
                    data_dir=eval_folder,
                    train_eval_flag="eval"),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=False
            )}

class CarlaH5Dataset_Simple(Dataset):
    def __init__(self, data_dir,
                 train_eval_flag="train", sequence_len=200):
        self.data_dir = data_dir
        self.data_list = glob.glob(data_dir+'*.h5')
        self.sequnece_len = sequence_len
        self.train_eval_flag = train_eval_flag

        self.build_transform()

    def build_transform(self):
        if self.train_eval_flag == "train" or self.train_eval_flag == "fine":
            self.transform = transforms.Compose([
                transforms.RandomOrder([
                    RandomTransWrapper(
                        seq=iaa.GaussianBlur(
                            (0, 1.5)),
                        p=0.1),
                    RandomTransWrapper(
                        seq=iaa.AdditiveGaussianNoise(
                            loc=0,
                            scale=(0.0, 0.05),
                            per_channel=0.5),
                        p=0.1),
                    RandomTransWrapper(
                        seq=iaa.Dropout(
                            (0.0, 0.10),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.CoarseDropout(
                            (0.0, 0.10),
                            size_percent=(0.08, 0.2),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Add(
                            (-20, 20),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Multiply(
                            (0.9, 1.1),
                            per_channel=0.2),
                        p=0.4),
                    RandomTransWrapper(
                        # seq=iaa.ContrastNormalization(
                        seq=iaa.LinearContrast(
                            (0.8, 1.2),
                            per_channel=0.5),
                        p=0.09),
                    ]),
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_idx = idx
        file_name = self.data_list[data_idx]

        with h5py.File(file_name, 'r') as h5_file:
            img_batch = np.array(h5_file['rgb'])
            img_batch = torch.cat([self.transform(out).unsqueeze(0) for out in img_batch], dim=0)

            target_batch = np.array(h5_file['targets'])
            target_batch = target_batch.astype(np.float32)

            speed_batch = np.array([target_batch[:, 5] / 40, ]).astype(np.float32)

            ''' target_vec list '''
            target_vec_list = []
            mask_vec_list = []
            weather_vec_list = []
            posi_list = []
            for i in range(self.sequnece_len):
                command_batch = int(target_batch[i][7]) - 2
                if command_batch == -2:
                    command_batch = 0
                target_vec_batch = np.zeros((4, 3), dtype=np.float32)
                target_vec_batch[command_batch, :] = target_batch[i][:3]
                mask_vec_batch = np.zeros((4, 3), dtype=np.float32)
                mask_vec_batch[command_batch, :] = 1

                target_vec_list.append(target_vec_batch.reshape(-1))
                mask_vec_list.append(mask_vec_batch.reshape(-1))

                ''' for weather '''
                weather_vec_list.append(int(target_batch[i][8]))

                ''' for position '''
                posi_x, posi_y = get_predicted_wheel_location(0, 0, target_batch[i][0], 0,
                                                              np.max(target_batch[i][5], 0) / 40, 0.1)
                posi_list.append([posi_x.astype(np.float32), posi_y.astype(np.float32)])

            # add preprocess
        return img_batch, speed_batch, np.array(target_vec_list), np.array(mask_vec_list), \
               np.array(weather_vec_list), np.array(posi_list)

def get_predicted_wheel_location(x, y, steering_angle, yaw, v, time_stamp=0.05):
    wheel_heading = np.deg2rad(yaw) + steering_angle
    # wheel_traveled_dis = v * (_current_timestamp - vars.t_previous)
    wheel_traveled_dis = v * time_stamp
    return [x + wheel_traveled_dis * np.cos(wheel_heading), y + wheel_traveled_dis * np.sin(wheel_heading)]

def main():
    train_dir = "/SSD2/datasets/carla/gen_data_simple/"
    eval_dir = "/SSD2/datasets/carla/gen_data_simple/"
    batch_size = 1
    workers = 1

    carla_data = CarlaH5Data_Simple(
        train_folder=train_dir,
        eval_folder=eval_dir,
        batch_size=batch_size,
        num_workers=workers
    )

    train_loader = carla_data.loaders["train"]
    eval_loader = carla_data.loaders["eval"]

    for i, (img, speed, target, mask, weather, weather0_mask,
            weather1_mask, weather2_mask, weather3_mask) in enumerate(train_loader):
        # imgs1 = img[:9]
        # save_image(imgs1, "save2.png", nrow=9)
        print(img)
        print(speed)
        print(target)
        print(mask)
        print(i)

if __name__ == '__main__':
    main()