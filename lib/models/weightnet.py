from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class WeightNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        self.N = cfg.WEIGHT.NUM_JOINTS       

        super(WeightNet, self).__init__()
        self.conv1 = nn.Conv2d(2*self.N, 2*self.N, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(2*self.N, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(2*self.N, 2*self.N, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(2*self.N, momentum=BN_MOMENTUM)
        self.avgpool = nn.AdaptiveAvgPool2d((1))

        self.fc1 = nn.Linear(5*self.N, 3*self.N)
        self.fc2 = nn.Linear(3*self.N, 3)


    def forward(self, l1, l2, l3, s, t):
        x = torch.cat((s,t),dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)

        x = torch.cat((x.squeeze(-1).squeeze(-1), l1, l2, l3),dim=1)

        x = self.fc1(x)
        x = self.fc2(x)

        out = torch.exp(x)
        return out

    def init_weights(self):
        logger.info('=>Weightnet init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.constant_(m.weight, 0)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.normal_(m.weight, std=0.001)

def get_weight_net(cfg, **kwargs):

    model = WeightNet(cfg, **kwargs)
    model.init_weights()

    return model
