import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import copy
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18_carla', 'resnet34_carla', 'resnet50_carla']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

model_paths = {
    'resnet34': 'models/resnet34-333f7ec4.pth',
}
'''
in_planes: input channels
out_planes: output channels
'''

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # AdaptiveAvgPooling2d는 Batch, Channel은 유지.
        # 원하는 output W, H를 입력하면 kernal_size, stride등을 자동으로 설정하여 연산.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention(3)

        self.downsample = downsample
        self.stride = stride

        self.attention = attention

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.attention:
            out = self.ca(out) * out
            out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention(3)

        self.downsample = downsample
        self.stride = stride

        self.attention = attention

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.attention:
            out = self.ca(out) * out
            out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, class_latent_size=128, content_latent_size=256):
        self.class_latent_size = class_latent_size
        self.content_latent_size = content_latent_size

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # latent vectors mu and sigma
        self.conv_output_size = [512, 8, 8]
        self.resnet_fc1 = nn.Linear(self.conv_output_size[0] * self.conv_output_size[1] * self.conv_output_size[2], 512)
        # self.resnet_fc1_bn = nn.BatchNorm1d(512)
        # self.resnet_fc2 = nn.Linear(512, 256)
        # self.resnet_fc2_bn = nn.BatchNorm1d(256)
        self.linear_mu = nn.Linear(512, self.content_latent_size)
        self.linear_logsigma = nn.Linear(512, self.content_latent_size)
        self.linear_classcode = nn.Linear(512, self.class_latent_size)

        # Sampling vector
        self.resnet_fc4 = nn.Linear(self.content_latent_size + self.class_latent_size, 512)
        # self.resnet_fc4_bn = nn.BatchNorm1d(512)
        self.resnet_fc5 = nn.Linear(512, self.conv_output_size[0] * self.conv_output_size[1] * self.conv_output_size[2])
        # self.resnet_fc5_bn = nn.BatchNorm1d(self.conv_output_size)

        # Decoder
        self.convTrans1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),
            # nn.BatchNorm2d(128, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),
            # nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),
            # nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),
            # nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid(),
        )

        self.convTrans5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),
            # nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid(),
        )
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            # nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid(),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
            # nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid(),
        )
        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            # nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def _make_layer(self, block, planes, blocks, stride=1, attention=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, attention))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # output size : batch, 512, 8, 8
        x = x.view(x.size(0), -1)
        x = self.resnet_fc1(x)
        # x = self.resnet_fc1_bn(self.resnet_fc1(x))
        # x = self.resnet_fc2_bn(self.resnet_fc2(x))

        mu = self.linear_mu(x)
        logsigma = self.linear_logsigma(x)
        classcode = self.linear_classcode(x)

        return mu, logsigma, classcode

    def decoder(self, z):
        # x = self.relu(self.resnet_fc4_bn(self.resnet_fc4(z)))
        # x = self.relu(self.resnet_fc5_bn(self.resnet_fc5(x))).view(-1, 512, 3, 7)
        x = self.relu(self.resnet_fc4(z))
        x = self.relu(self.resnet_fc5(x)).view(-1, self.conv_output_size[0], self.conv_output_size[1], self.conv_output_size[2])
        x = self.convTrans1(x)
        x = self.convTrans2(x)
        x = self.convTrans3(x)
        x = self.convTrans4(x)
        x = F.interpolate(x, size=(256, 256))

        return x

def resnet18_carla(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet34_carla(pretrained=False, class_latent_size=128, content_latent_size=256, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], class_latent_size, content_latent_size, **kwargs)
    if pretrained:
        # pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        pretrained_state_dict = torch.load(model_paths['resnet34'])
        now_state_dict        = model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in now_state_dict}
        # 2. overwrite entries in the existing state dict
        now_state_dict.update(pretrained_state_dict)
        # 3. load the new state dict
        model.load_state_dict(now_state_dict)
    return model


def resnet50_carla(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model
