''' input size가 lusr은 128 x 128 인데 256 x 256을 사용할 것이기 때문에
관련 feature의 dimension 들을 2배로

210818
resnet50을 써보자
'''

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
# from utils.grad_reverse import grad_reverse
import torch.utils.model_zoo as model_zoo

# Models for CARLA autonomous driving
''' ResNet Encoder Start '''
__all__ = ['ResNet', 'resnet18_carla', 'resnet34_carla', 'resnet50_carla']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

model_paths = {
    'resnet34': '/SSD1/models/resnet34-333f7ec4.pth',
    'resnet50': '/SSD1/models/resnet50-19c8e357.pth',
}

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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
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

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
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

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers):

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

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # fc layers
        self.img_fc = nn.Sequential(
            nn.Linear(131072, 2048), nn.Dropout(0.3), nn.ReLU(),
            nn.Linear(2048, 512), nn.Dropout(0.3), nn.ReLU(),
        )

        self.speed_fc = nn.Sequential(
            nn.Linear(1, 128), nn.Dropout(0.5), nn.ReLU(),
            nn.Linear(128, 128), nn.Dropout(0.5), nn.ReLU(),
        )

        self.emb_fc = nn.Sequential(
            nn.Linear(512 + 128, 512), nn.Dropout(0.5), nn.ReLU(),
        )

        self.speed_branch = nn.Sequential(
            nn.Linear(512, 256), nn.Dropout(0.5), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.posi_branch = nn.Sequential(
            nn.Linear(512, 256), nn.Dropout(0.5), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2),
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

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, speed):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feature = self.layer2(x)
        x = self.layer3(feature)
        x = self.layer4(x)  # output size : batch, 512, 8, 8
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.img_fc(x)

        speed = self.speed_fc(speed)
        emb = torch.cat([x, speed], dim=1)
        emb = self.emb_fc(emb)

        pred_speed = self.speed_branch(x)
        pred_posi = self.posi_branch(x)

        return emb, pred_speed, pred_posi, feature

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

def resnet34_carla(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
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
        # pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        pretrained_state_dict = torch.load(model_paths['resnet50'])
        now_state_dict        = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in now_state_dict}
        # 2. overwrite entries in the existing state dict
        now_state_dict.update(pretrained_state_dict)
        # 3. load the new state dict
        model.load_state_dict(now_state_dict)
    return model
''' ResNet Encoder End'''

def reparameterize(mu, logsigma):
    std = torch.exp(0.5*logsigma)
    eps = torch.randn_like(std)
    return mu + eps*std

class WeatClassifier(nn.Module):
    def __init__(self, input_latent_size):
        super(WeatClassifier, self).__init__()
        self.weat_classifier = nn.Sequential(
            nn.Linear(input_latent_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.weat_classifier(x)

class CarlaActionPredictor(nn.Module):
    def __init__(self, content_latent_size=64):
        super(CarlaActionPredictor, self).__init__()

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512 + content_latent_size, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                # nn.Dropout(self.dropout_vec[i*2+14]),
                nn.ReLU(),
                nn.Linear(256, 3),
            ) for i in range(4)
        ])

    def forward(self, emb):
        output = torch.cat([out(emb) for out in self.branches], dim=1)

        return output

class MIDiscriminator(nn.Module):
    def __init__(self, content_latent_size=64, emb_size=512):
        super(MIDiscriminator, self).__init__()

        self.mi_discriminator = torch.nn.Sequential(
            torch.nn.Linear(emb_size + content_latent_size * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.mi_discriminator(x)

class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(512, 64, kernel_size=3)
        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.l0 = nn.Linear(25088 + 768, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = h.contiguous()
        h = h.view(y.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)

class LocalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(512 + 768, 512, kernel_size=1)
        self.c1 = nn.Conv2d(512, 512, kernel_size=1)
        self.c2 = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)

class PriorDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(768, 1024)
        self.l1 = nn.Linear(1024, 256)
        self.l2 = nn.Linear(256, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class GatingFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.gating = torch.nn.Sequential(
            nn.Linear(512 + 128, 256), nn.ReLU(),
            nn.Linear(256, 2), nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.gating(x)

class CarlaDisentangled(nn.Module):
    def __init__(self, encoder_name='basic', class_latent_size=64, content_latent_size=128):
        super(CarlaDisentangled, self).__init__()
        self.class_latent_size = class_latent_size
        self.content_latent_size = content_latent_size
        self.weat_classifier = WeatClassifier(self.class_latent_size) # content_latent_size: pair 정보의 latent size

        ''' MI '''
        self.midiscriminator = MIDiscriminator(content_latent_size, 512)
        self.local_d = LocalDiscriminator()
        self.global_d = GlobalDiscriminator()
        self.prior_d = PriorDiscriminator()

        self.encoder = resnet50_carla(True)
        ''' gating function '''
        self.gating_function = GatingFunction()

        ''' Action Prediction '''
        self.action_predictor1 = CarlaActionPredictor(self.content_latent_size)
        self.action_predictor2 = CarlaActionPredictor(self.content_latent_size)
        self.action_predictor3 = CarlaActionPredictor(self.content_latent_size)
        self.action_predictor4 = CarlaActionPredictor(self.content_latent_size)

    def MI_minimization(self, emb, contentcode):
        emb_prime = torch.cat((emb[1:], emb[0].unsqueeze(0)), dim=0)
        mi_real = self.midiscriminator(torch.cat([contentcode, emb_prime], dim=1))
        mi_fake = self.midiscriminator(torch.cat([contentcode, emb], dim=1))

        return mi_real, mi_fake

    def deep_infomax_loss(self, y, M):
        M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand(-1, -1, 32, 32)

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej_l = -F.softplus(-self.local_d(y_M)) #.mean()
        Em_l = F.softplus(self.local_d(y_M_prime)) #.mean()
        # LOCAL = (Em - Ej) * self.beta

        Ej_g = -F.softplus(-self.global_d(y, M)) #.mean()  # y: E_psi(x), M: x, x': M_prime in the paper
        Em_g = F.softplus(self.global_d(y, M_prime)) #.mean()
        # GLOBAL = (Em - Ej) * self.alpha

        prior = torch.rand_like(y)

        eps = torch.finfo(float).eps
        term_a = torch.log(self.prior_d(prior) + eps) #.mean()
        term_b = torch.log(1.0 - self.prior_d(y) + eps) #.mean()
        # PRIOR = - (term_a + term_b) * self.gamma

        # return LOCAL + GLOBAL + PRIOR
        return Ej_l, Em_l, Ej_g, Em_g, term_a, term_b

    def forward(self, img, speed, mu, logsigma, classcode):
        '''
            lane pair로 학습한 Vae 면 classcode가 날씨
            weather pair로 학습한 vae 면 mu가 날씨
            즉, content가 pair 이 된다는 의미
        '''

        emb, pred_speed, pred_posi, M = self.encoder(img, speed)
        # M : 100, 128, 32, 32
        ''' MINE '''
        # # contentcode = reparameterize(mu, logsigma)
        # ### minimization
        # mi_real, mi_fake = self.MI_minimization(emb, torch.cat([mu, classcode], dim=1))
        # ### maximization
        # Ej_l, Em_l, Ej_g, Em_g, term_a, term_b = self.deep_infomax_loss(torch.cat([mu, classcode, emb], dim=1), M)

        ''' Weather branch '''
        weat_out = self.weat_classifier(classcode)

        ''' gating '''
        weights = self.gating_function(torch.cat([emb, mu], dim=1))

        emb_g = weights[:, 0].unsqueeze(1) * emb
        mu_g = weights[:, 1].unsqueeze(1) * mu

        ''' Actions Predictor '''
        act_output1 = self.action_predictor1(torch.cat([emb_g, mu_g], dim=1))
        act_output2 = self.action_predictor2(torch.cat([emb_g, mu_g], dim=1))
        act_output3 = self.action_predictor3(torch.cat([emb_g, mu_g], dim=1))
        act_output4 = self.action_predictor4(torch.cat([emb_g, mu_g], dim=1))

        # return weat_out, pred_speed, pred_posi,\
        #     act_output1, act_output2, act_output3, act_output4, \
        #     mi_real, mi_fake, Ej_l, Em_l, Ej_g, Em_g, term_a, term_b
        return weat_out, pred_speed, \
               act_output1, act_output2, act_output3, act_output4


































