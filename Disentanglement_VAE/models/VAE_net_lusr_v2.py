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
from utils.utils import reparameterize
from models.resnet_decoder import ResNet34Dec
from models.resnet_vae import resnet34
from models.resnet18_vae import ResNet18Enc, ResNet18Dec
from models.vgg_vae import vgg16

class CarlaDecoderResNet(nn.Module):
    def __init__(self, latent_size=32 + 64, output_channel=3):
        super(CarlaDecoderResNet, self).__init__()
        self.fc = nn.Linear(latent_size, 1024)
        self.fc2 = nn.Linear(1024, 32768)
        self.relu = nn.ReLU(inplace=True)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = self.relu(self.fc2(x))
        # x = torch.reshape(x, (-1, 256, 6, 6))
        x = torch.reshape(x, (-1, 512, 8, 8))
        x = self.main(x)
        return x

# Models for CARLA autonomous driving
class CarlaEncoder(nn.Module):
    def __init__(self, class_latent_size=32, content_latent_size=64, input_channel=3, flatten_size=18432):
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


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)

class VQVAE_Encoder(nn.Module):
    def __init__(self, class_latent_size=512, content_latent_size=512, input_channel=3, flatten_size=131072):
        super(VQVAE_Encoder, self).__init__()
        self.class_latent_size = class_latent_size
        self.content_latent_size = content_latent_size
        self.flatten_size = flatten_size
        d = 128
        bn=True
        self.main = nn.Sequential(
            nn.Conv2d(3, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
        )

        self.linear_mu = nn.Linear(flatten_size, content_latent_size)
        self.linear_logsigma = nn.Linear(flatten_size, content_latent_size)  # content_latent_size: 16
        self.linear_classcode = nn.Linear(flatten_size, class_latent_size)  # class_latent_size: 8

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        logsigma = self.linear_logsigma(x)
        classcode = self.linear_classcode(x)

        return mu, logsigma, classcode

    def get_feature(self, x):
        mu, logsigma, classcode = self.forward(x)
        return mu


class VQVAE_Decoder(nn.Module):
    def __init__(self, latent_size=512 + 512, output_channel=3):
        super(VQVAE_Decoder, self).__init__()
        self.fc = nn.Linear(latent_size, 131072)
        d = 128

        self.main = nn.Sequential(
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                d, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        # x = torch.reshape(x, (-1, 256, 6, 6))
        x = torch.reshape(x, (-1, 128, 32, 32))
        x = self.main(x)
        return x


class CarlaDisentangledVAE(nn.Module):
    def __init__(self, class_latent_size=32, content_latent_size=64, img_channel=3, flatten_size=18432,
                 net_type='Basic', pretrained = True):
        super(CarlaDisentangledVAE, self).__init__()

        if net_type == 'Basic':
            self.encoder = CarlaEncoder(class_latent_size, content_latent_size, img_channel, flatten_size)
            self.decoder = CarlaDecoder(class_latent_size + content_latent_size, img_channel)
        elif net_type == 'ResNet34':
            self.encoder = resnet34(pretrained)
            # self.decoder = CarlaDecoderResNet(class_latent_size + content_latent_size, img_channel)
            self.decoder = CarlaDecoder(class_latent_size + content_latent_size, img_channel)
        elif net_type == 'ResNet18':
            self.encoder = ResNet18Enc(class_latent_size=class_latent_size, content_latent_size=content_latent_size)
            self.decoder = ResNet18Dec(class_latent_size=class_latent_size, content_latent_size=content_latent_size)
        elif net_type == 'vgg16':
            self.encoder = vgg16()
            self.decoder = CarlaDecoder(class_latent_size + content_latent_size, img_channel)
        elif net_type == 'vqvae':
            self.encoder = VQVAE_Encoder()
            self.decoder = VQVAE_Decoder()

    def vae_loss(self, x, mu, logsigma, recon_x, beta=1):
        recon_loss = F.mse_loss(x, recon_x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
        kl_loss = kl_loss / torch.numel(x)
        return recon_loss + kl_loss * beta

    def forward_loss(self, x, beta):
        mu, logsigma, classcode = self.encoder(x)
        contentcode = reparameterize(mu, logsigma)
        shuffled_classcode = classcode[torch.randperm(classcode.shape[0])]

        latentcode1 = torch.cat([contentcode, shuffled_classcode], dim=1)
        latentcode2 = torch.cat([contentcode, classcode], dim=1)

        recon_x1 = self.decoder(latentcode1)
        recon_x2 = self.decoder(latentcode2)

        return self.vae_loss(x, mu, logsigma, recon_x1, beta) + self.vae_loss(x, mu, logsigma, recon_x2, beta)

    def backward_loss(self, x):
        mu, logsigma, classcode = self.encoder(x)
        shuffled_classcode = classcode[torch.randperm(classcode.shape[0])]
        # randcontent = torch.randn_like(mu).to(device)
        randcontent = torch.randn_like(mu)

        latentcode1 = torch.cat([randcontent, classcode], dim=1)
        latentcode2 = torch.cat([randcontent, shuffled_classcode], dim=1)

        recon_imgs1 = self.decoder(latentcode1).detach()
        recon_imgs2 = self.decoder(latentcode2).detach()

        cycle_mu1, cycle_logsigma1, cycle_classcode1 = self.encoder(recon_imgs1)
        cycle_mu2, cycle_logsigma2, cycle_classcode2 = self.encoder(recon_imgs2)

        cycle_contentcode1 = reparameterize(cycle_mu1, cycle_logsigma1)
        cycle_contentcode2 = reparameterize(cycle_mu2, cycle_logsigma2)

        bloss = F.l1_loss(cycle_contentcode1, cycle_contentcode2)
        return bloss

    def forward(self, img1, img2, img3, img4, beta, mode=0, latentcode = None):
        if mode == 0:
            floss1 = self.forward_loss(img1, beta)
            floss2 = self.forward_loss(img2, beta)
            floss3 = self.forward_loss(img3, beta)
            floss4 = self.forward_loss(img4, beta)

            img_cat = torch.cat([img1, img2, img3, img4], dim=0)
            bloss = self.backward_loss(img_cat)

            return (floss1+floss2+floss3+floss4) / 4, bloss

        elif mode == 1:
            return self.encoder(img1)

        elif mode == 2:
            return self.decoder(latentcode)






















