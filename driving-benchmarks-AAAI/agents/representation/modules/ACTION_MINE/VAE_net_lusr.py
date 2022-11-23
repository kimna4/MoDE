''' input size가 lusr은 128 x 128 인데 256 x 256을 사용할 것이기 때문에
관련 feature의 dimension 들을 2배로
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Beta
from torch.autograd import Function
import torch.utils.model_zoo as model_zoo

def reparameterize(mu, logsigma):
    std = torch.exp(0.5*logsigma)
    eps = torch.randn_like(std)
    return mu + eps*std

# Models for CARLA autonomous driving
class CarlaEncoder(nn.Module):
    def __init__(self, class_latent_size=64, content_latent_size=128, input_channel=3, flatten_size=18432):
        super(CarlaEncoder, self).__init__()
        self.class_latent_size = class_latent_size
        self.content_latent_size = content_latent_size
        self.flatten_size = flatten_size

        self.main = nn.Sequential(
            nn.Conv2d(input_channel, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(),
            nn.Conv2d(256, 512, 4, stride=2), nn.ReLU()
        )

        self.linear_mu = nn.Linear(flatten_size, content_latent_size)
        self.linear_logsigma = nn.Linear(flatten_size, content_latent_size)  # content_latent_size: 16
        self.linear_classcode = nn.Linear(flatten_size, class_latent_size)  # class_latent_size: 8

    def forward(self, x):
        x = self.main(x)
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        logsigma = self.linear_logsigma(x)
        classcode = self.linear_classcode(x)

        return mu, logsigma, classcode

    def get_feature(self, x):
        mu, logsigma, classcode = self.forward(x)
        return mu


class CarlaDecoder(nn.Module):
    def __init__(self, latent_size=32 + 64, output_channel=3):
        super(CarlaDecoder, self).__init__()
        self.fc = nn.Linear(latent_size, 18432)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        # x = torch.reshape(x, (-1, 256, 6, 6))
        x = torch.reshape(x, (-1, 512, 6, 6))
        x = self.main(x)
        return x

class CarlaDisentangledVAE(nn.Module):
    def __init__(self, class_latent_size=32, content_latent_size=64, img_channel=3, flatten_size=18432,
                 net_type='Basic', pretrained = True):
        super(CarlaDisentangledVAE, self).__init__()

        self.encoder = CarlaEncoder(class_latent_size, content_latent_size, img_channel, flatten_size)
        # self.decoder = CarlaDecoder(class_latent_size + content_latent_size, img_channel)

    def forward(self, x):
        mu, logsigma, classcode = self.encoder(x)
        contentcode = reparameterize(mu, logsigma)
        latentcode = torch.cat([contentcode, classcode], dim=1)

        # recon_x = self.decoder(latentcode)

        return mu, logsigma, classcode #, recon_x



