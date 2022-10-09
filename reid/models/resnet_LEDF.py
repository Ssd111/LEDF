from __future__ import absolute_import

import copy

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import numpy as np
from torch.autograd import Variable
import random
import math
import torchvision.transforms.functional as FF
from reid.utils.data import transforms as T

import torchvision.transforms as standard_transforms

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50_ledf', 'resnet101',
           'resnet152']

normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,args=None):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=pretrained)
        self.network = resnet
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.args = args
        self.pecent = 1./3
        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
        if not pretrained:
            self.reset_params()

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x,epoch=0,memory=None,t0=None,ledf=False,gt=None):
        # lam0 = 0.5
        if self.args!=None:
            lam0 = self.args.lam0
        else:
            lam0 = 0.5
        # print(lam0)


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        if ledf:
            if t0 == 0:
                t1 = 1
                t2 = 2
            elif t0 == 1:
                t1 = 0
                t2 = 2
            else:
                t1 = 0
                t2 = 1

            interval = 10
            if epoch % interval == 0:
                self.pecent = 3.0 / 10 + (epoch / interval) * 2.0 / 10

            self.eval()
            x_new = x.clone().detach()
            x_new = Variable(x_new.data, requires_grad=True)
            x_new_view = self.layer1(x_new)
            x_new_view = self.layer2(x_new_view)
            x_new_view = self.layer3(x_new_view)
            x_new_view = self.layer4(x_new_view)
            x_new_view = self.gap(x_new_view)
            x_new_view = x_new_view.view(x_new_view.size(0), -1)
            x_new_view = self.feat_bn(x_new_view)
            x_new_view = F.normalize(x_new_view, dim=1)




            x_new2 = x.clone().detach()
            x_new2 = Variable(x_new2.data, requires_grad=True)
            x_new_view2 = self.layer1(x_new2)
            x_new_view2 = self.layer2(x_new_view2)
            x_new_view2 = self.layer3(x_new_view2)
            x_new_view2 = self.layer4(x_new_view2)
            x_new_view2 = self.gap(x_new_view2)
            x_new_view2 = x_new_view2.view(x_new_view2.size(0), -1)
            x_new_view2 = self.feat_bn(x_new_view2)
            x_new_view2 = F.normalize(x_new_view2,dim=1)




            self.zero_grad()


            output = x_new_view2.mm(memory[t0].module.features.to(x_new.device).t())
            output = output / 0.05
            loss_s0 = F.cross_entropy(output, gt)
            loss_s0.backward()
            grads_val2 = x_new2.grad.clone().detach()


            class_num = output.shape[1]
            index = gt
            num_rois = x_new.shape[0]
            num_channel = x_new.shape[1]
            H = x_new.shape[2]
            W = x_new.shape[3]
            HW = x_new.shape[2] * x_new.shape[3]


            features2 = torch.cat((memory[t0].module.features,memory[t1].module.features,memory[t2].module.features),0)
            sim = x_new_view.mm(features2.to(x_new.device).t())
            sim = sim / 0.05
            loss_s1 = F.cross_entropy(sim, gt)
            self.zero_grad()
            loss_s1.backward()
            grads_val = x_new.grad.clone().detach()
            grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
            channel_mean = grad_channel_mean
            grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
            spatial_mean = torch.sum(x_new * grad_channel_mean, 1)
            spatial_mean = spatial_mean.view(num_rois, HW)




            grad_channel_mean2 = torch.mean(grads_val2.view(num_rois, num_channel, -1), dim=2)
            channel_mean2 = grad_channel_mean2
            grad_channel_mean2 = grad_channel_mean2.view(num_rois, num_channel, 1, 1)


            spatial_mean2 = torch.sum(x_new2 * grad_channel_mean2, 1)
            spatial_mean2 = spatial_mean2.view(num_rois, HW)

            choose_one = random.randint(0, 9)
            if choose_one <= 4:
                # ---------------------------- spatial -----------------------

                a1 = spatial_mean.argsort(dim=1, descending=False)
                a2 = spatial_mean2.argsort(dim=1, descending=False)
                a11 = a1.argsort(dim=1,descending=False)
                a22 = a2.argsort(dim=1,descending=False)
                a3 = a11 - a22
                vector_thresh_percent = math.ceil(a3.size(1) * lam0)
                vector_thresh_value = torch.sort(a3, dim=0, descending=True)[0][:,vector_thresh_percent]
                vector_thresh_value = vector_thresh_value.view(num_rois, 1).expand(num_rois, HW)

                mask_all = torch.where(a3 > vector_thresh_value,torch.zeros(spatial_mean.shape).cuda(),
                                            torch.ones(spatial_mean.shape).cuda())
                mask_all = mask_all.reshape(num_rois, H, W).view(num_rois, 1, H, W)

            else:
                # -------------------------- channel ----------------------------
                a1 = channel_mean.argsort(dim=1, descending=False)
                a2 = channel_mean2.argsort(dim=1, descending=False)
                a11 = a1.argsort(dim=1,descending=False)
                a22 = a2.argsort(dim=1,descending=False)
                a3 = a11 - a22


                vector_thresh_percent = math.ceil(a3.size(1) * lam0)
                vector_thresh_value = torch.sort(a3, dim=0, descending=True)[0][:,vector_thresh_percent]
                vector_thresh_value = vector_thresh_value.view(num_rois, 1).expand(num_rois, num_channel)

                mask_all = torch.where(a3 > vector_thresh_value,
                                       torch.zeros(channel_mean.shape).cuda(),
                                       torch.ones(channel_mean.shape).cuda())
                mask_all = mask_all.view(num_rois, num_channel, 1, 1)

            self.train()
            mask_all = Variable(mask_all, requires_grad=True)
            x = x * mask_all

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gap(x)

        x = x.view(x.size(0), -1)


        if self.cut_at_pooling:
            return x
        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)


        tri_features = x
        if self.training is False:
            bn_x = F.normalize(bn_x)
            return bn_x


        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        return bn_x, tri_features


    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def get_weights(self):
        weights = []
        for p in self.network.parameters():
            weights.append(p.data.clone().flatten())
        return torch.cat(weights)




def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50_ledf(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)

