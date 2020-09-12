import torch
from  torch import nn
import  torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

from models.backbone.resnet import *
from models.AATM import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

def init_pretrained_weight(model, model_url):
    """Initializes model with pretrained weight

    Layers that don't match with pretrained layers in name or size are kept unchanged
    """
    pretrain_dict = model_zoo.load_url(model_url, model_dir='./')
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weight_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class STAM(nn.Module):

    def __init__(self, num_classes, model_name, pretrain_choice,seq_len, dropout=0, layer_num=2, feature_method='cat',
                 spatial_method = 'avg',
                 temporal_method = 'avg',
                 is_mutual_channel_attention='yes',
                 is_mutual_spatial_attention='yes',
                 is_appearance_channel_attention='yes',
                 is_appearance_spatial_attention='yes',
                 is_down_channel = 'yes'):
        super(STAM, self).__init__()

        self.in_planes = 2048
        self.base = ResNet()

        if pretrain_choice == 'imagenet':
            init_pretrained_weight(self.base, model_urls[model_name])
            print('Loading pretrained ImageNet model ......')

        self.seq_len = seq_len
        self.num_classes = num_classes
        self.plances = 1024
        self.mid_channel = int(self.plances * 0.5)
        self.dropout = dropout
        self.layer_num = layer_num
        self.feature_method = feature_method
        self.spatial_method = spatial_method
        self.temporal_method = temporal_method
        self.is_down_channel = is_down_channel

        self.avg_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        if self.is_down_channel :
            self.down_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_planes, out_channels=self.plances,kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.plances),
                self.relu
            )
        else:
            self.plances = 2048

        if self.feature_method == 'cat':
            self.cat_conv = nn.Conv1d(in_channels=self.layer_num, out_channels=1, kernel_size=1)
            # self.cat_conv.apply(weights_init_kaiming)

        if self.layer_num == 3:
            t = seq_len
            self.layer1 = AATM(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=t / 2, spatial_method=self.spatial_method,
                               is_mutual_channel_attention=is_mutual_channel_attention,
                               is_mutual_spatial_attention=is_mutual_spatial_attention,
                               is_appearance_channel_attention=is_appearance_channel_attention,
                               is_appearance_spatial_attention=is_appearance_spatial_attention)
            t = t / 2
            self.layer2 = AATM(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=t / 2, spatial_method=self.spatial_method,
                               is_mutual_channel_attention=is_mutual_channel_attention,
                               is_mutual_spatial_attention=is_mutual_spatial_attention,
                               is_appearance_channel_attention=is_appearance_channel_attention,
                               is_appearance_spatial_attention=is_appearance_spatial_attention)
            t = t / 2
            self.layer3 = AATM(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=t / 2, spatial_method=self.spatial_method,
                               is_mutual_channel_attention=is_mutual_channel_attention,
                               is_mutual_spatial_attention=is_mutual_spatial_attention,
                               is_appearance_channel_attention=is_appearance_channel_attention,
                               is_appearance_spatial_attention=is_appearance_spatial_attention)

        elif self.layer_num == 2:

            t = seq_len
            self.layer1 = AATM(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=t / 2, spatial_method=self.spatial_method,
                               is_mutual_channel_attention=is_mutual_channel_attention,
                               is_mutual_spatial_attention=is_mutual_spatial_attention,
                               is_appearance_channel_attention=is_appearance_channel_attention,
                               is_appearance_spatial_attention=is_appearance_spatial_attention)
            t = t / 2
            self.layer2 = AATM(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=t / 2, spatial_method=self.spatial_method,
                               is_mutual_channel_attention=is_mutual_channel_attention,
                               is_mutual_spatial_attention=is_mutual_spatial_attention,
                               is_appearance_channel_attention=is_appearance_channel_attention,
                               is_appearance_spatial_attention=is_appearance_spatial_attention)

        elif self.layer_num == 1:

            t = seq_len
            self.layer1 = AATM(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=t / 2, spatial_method=self.spatial_method,
                               is_mutual_channel_attention=is_mutual_channel_attention,
                               is_mutual_spatial_attention=is_mutual_spatial_attention,
                               is_appearance_channel_attention=is_appearance_channel_attention,
                               is_appearance_spatial_attention=is_appearance_spatial_attention)

        self.bottleneck = nn.BatchNorm1d(self.plances)
        self.classifier = nn.Linear(self.plances, self.num_classes, bias=False)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weight_init_classifier)

    def aggregate_feature(self, feature_list):

        num = len(feature_list)

        if self.feature_method == 'cat' :

            cat_feature = torch.stack(feature_list, 1)
            feature = self.cat_conv(cat_feature)
            feature = self.sigmoid(feature).view(feature.size(0), -1)

        elif self.feature_method == 'final':

            feature = feature_list[num]

        return feature

    def forward(self, x, pids=None, camid=None):

        b, t, c, w, h = x.size()
        x = x.view(b * t, c, w, h)
        feat_map = self.base(x)  # (b * t, c, 16, 8)
        w = feat_map.size(2)
        h = feat_map.size(3)
        feat_map = self.down_channel(feat_map)

        feat_map = feat_map.view(b, t, -1, w, h)
        if self.layer_num == 3 :

            list = []
            feat_map_1, feature_1 = self.layer1(feat_map)
            list.append(feature_1)
            feat_map_2, feature_2 = self.layer2(feat_map_1)
            list.append(feature_2)
            feat_map_3, feature_3 = self.layer3(feat_map_2)
            list.append(feature_3)

            feature = self.aggregate_feature(list)

        elif self.layer_num == 2:

            list = []
            feat_map_1, feature_1 = self.layer1(feat_map)
            list.append(feature_1)
            feat_map_2, feature_2 = self.layer2(feat_map_1)
            list.append(feature_2)

            feature = self.aggregate_feature(list)

        else:

            feat_map_1, feature_1 = self.layer1(feat_map)
            feature = feature_1

        feature_list = []
        feature_list.append(feature)
        BN_feature = self.bottleneck(feature)

        cls_score_list = []
        torch.cuda.empty_cache()

        if self.training:
            cls_score_list.append(self.classifier(BN_feature))
            return cls_score_list, BN_feature
        else:
            return BN_feature, pids, camid