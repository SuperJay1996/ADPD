from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_activation_layer, constant_init,
                      normal_init)
from torch.nn.modules.batchnorm import _BatchNorm


from .utils import channel_shuffle, load_checkpoint, make_divisible

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class ShuffleUnit(nn.Module):
    """ShuffleUnit block.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 groups=3,
                 first_block=True,
                 combine='add',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        act_cfg = copy.deepcopy(act_cfg)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_block = first_block
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4
        self.with_cp = with_cp

        if self.combine == 'add':
            self.depthwise_stride = 1
            self._combine_func = self._add
            assert in_channels == out_channels, (
                'in_channels must be equal to out_channels when combine '
                'is add')
        elif self.combine == 'concat':
            self.depthwise_stride = 2
            self._combine_func = self._concat
            self.out_channels -= self.in_channels
            self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f'Cannot combine tensors with {self.combine}. '
                             'Only "add" and "concat" are supported')

        self.first_1x1_groups = 1 if first_block else self.groups
        self.g_conv_1x1_compress = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.bottleneck_channels,
            kernel_size=1,
            groups=self.first_1x1_groups,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.depthwise_conv3x3_bn = ConvModule(
            in_channels=self.bottleneck_channels,
            out_channels=self.bottleneck_channels,
            kernel_size=3,
            stride=self.depthwise_stride,
            padding=1,
            groups=self.bottleneck_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.g_conv_1x1_expand = ConvModule(
            in_channels=self.bottleneck_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            groups=self.groups,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.act = build_activation_layer(act_cfg)

    @staticmethod
    def _add(x, out):
        # residual connection
        return x + out

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):

        def _inner_forward(x):
            residual = x

            out = self.g_conv_1x1_compress(x)
            out = self.depthwise_conv3x3_bn(out)

            if self.groups > 1:
                out = channel_shuffle(out, self.groups)

            out = self.g_conv_1x1_expand(out)

            if self.combine == 'concat':
                residual = self.avgpool(residual)
                out = self.act(out)
                out = self._combine_func(residual, out)
            else:
                out = self._combine_func(residual, out)
                out = self.act(out)
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class ShuffleNetV1(nn.Module):
    """ShuffleNetV1 backbone.
    """
    def __init__(self,
                 cfg,
                 out_indices=(2, 1, 0),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 with_cp=False):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        act_cfg = copy.deepcopy(act_cfg)
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super().__init__()
        self.stage_blocks = [4, 8, 4]
        self.groups = cfg['MODEL']['EXTRA']['GROUP']
        self.widen_factor = cfg['MODEL']['EXTRA']['WIDEN_FACTOR']

        for index in out_indices:
            if index not in range(0, 3):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 3). But received {index}')

        if frozen_stages not in range(-1, 3):
            raise ValueError('frozen_stages must be in range(-1, 3). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        if self.groups == 1:
            channels = (144, 288, 576)
        elif self.groups == 2:
            channels = (200, 400, 800)
        elif self.groups == 3:
            channels = (240, 480, 960)
        elif self.groups == 4:
            channels = (272, 544, 1088)
        elif self.groups == 8:
            channels = (384, 768, 1536)
        else:
            raise ValueError(f'{self.groups} groups is not supported for 1x1 '
                             'Grouped Convolutions')

        channels = [make_divisible(ch * self.widen_factor, 8) for ch in channels]

        self.in_channels = int(24 * self.widen_factor)

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(self.stage_blocks):
            first_block = (i == 0)
            layer = self.make_layer(channels[i], num_blocks, first_block)
            self.layers.append(layer)
        
        self.inplanes = 960
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )
    
    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(self.frozen_stages):
            layer = self.layers[i]
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=False)
            logger.info('=> init weights from {}'.format(pretrained))
        elif pretrained is None:
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d):
                    if 'conv1' in name:
                        normal_init(m, mean=0, std=0.01)
                    else:
                        normal_init(m, mean=0, std=1.0 / m.weight.shape[1])
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m.weight, val=1, bias=0.0001)
                    if isinstance(m, _BatchNorm):
                        if m.running_mean is not None:
                            nn.init.constant_(m.running_mean, 0)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)

    def make_layer(self, out_channels, num_blocks, first_block=False):
        """Stack ShuffleUnit blocks to make a layer.

        Args:
            out_channels (int): out_channels of the block.
            num_blocks (int): Number of blocks.
            first_block (bool, optional): Whether is the first ShuffleUnit of a
                sequential ShuffleUnits. Default: False, which means not using
                the grouped 1x1 convolution.
        """
        layers = []
        for i in range(num_blocks):
            first_block = first_block if i == 0 else False
            combine_mode = 'concat' if i == 0 else 'add'
            layers.append(
                ShuffleUnit(
                    self.in_channels,
                    out_channels,
                    groups=self.groups,
                    first_block=first_block,
                    combine=combine_mode,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
        
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    
def get_pose_net(cfg, is_train, **kwargs):

    model = ShuffleNetV1(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
