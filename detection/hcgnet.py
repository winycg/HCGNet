import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import logging

import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, dilation=1, conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(BasicConv, self).__init__()
        self.norm_name, norm = build_norm_layer(norm_cfg, in_channels, postfix=1)
        self.add_module(self.norm_name, norm)
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=False)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x, is_return=False):
        x = self.norm(x)
        post_x = F.relu(x)
        x = self.conv(post_x)
        if is_return:
            return post_x, x
        else:
            return x


class _SMG(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, groups=4, reduction_factor=2, forget_factor=2):
        super(_SMG, self).__init__()
        self.in_channels = in_channels
        self.reduction_factor = reduction_factor
        self.forget_factor = forget_factor
        self.growth_rate = growth_rate
        self.conv1_1x1 = BasicConv(in_channels, int(bn_size * growth_rate), kernel_size=1, stride=1)
        self.conv2_3x3 = BasicConv(int(bn_size * growth_rate), growth_rate, kernel_size=3, stride=1,
                                   padding=1, groups=groups)
        # Mobile
        self.conv_3x3 = BasicConv(growth_rate, growth_rate, kernel_size=3,
                                  stride=1, padding=1, groups=growth_rate,)
        self.conv_5x5 = BasicConv(growth_rate, growth_rate, kernel_size=3,
                                  stride=1, padding=2, groups=growth_rate, dilation=2)

        # GTSK layers
        self.global_context3x3 = build_conv_layer(None, growth_rate, 1, kernel_size=1)
        self.global_context5x5 = build_conv_layer(None, growth_rate, 1, kernel_size=1)

        self.fcall = build_conv_layer(None, 2 * growth_rate, 2 * growth_rate // self.reduction_factor, kernel_size=1)
        self.bn_attention = nn.BatchNorm1d(2 * growth_rate // self.reduction_factor)
        self.fc3x3 = build_conv_layer(None, 2 * growth_rate // self.reduction_factor, growth_rate, kernel_size=1)
        self.fc5x5 = build_conv_layer(None, 2 * growth_rate // self.reduction_factor, growth_rate, kernel_size=1)

        # SE layers
        self.global_forget_context = build_conv_layer(None, growth_rate, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn_forget = nn.BatchNorm1d(growth_rate // self.forget_factor)
        self.fc1 = build_conv_layer(None, growth_rate, growth_rate // self.forget_factor, kernel_size=1)
        self.fc2 = build_conv_layer(None, growth_rate // self.forget_factor, growth_rate, kernel_size=1)

    def forward(self, x):
        x_dense = x
        x = self.conv1_1x1(x)
        x = self.conv2_3x3(x)

        H, W = x.size(2), x.size(3)
        C = x.size(1)
        x_shortcut = x
        forget_context_weight = self.global_forget_context(x_shortcut)
        forget_context_weight = torch.flatten(forget_context_weight, start_dim=1)
        forget_context_weight = F.softmax(forget_context_weight, 1).reshape(-1, 1, H, W)
        x_shortcut_weight = self.global_pool(x_shortcut * forget_context_weight) * H * W
        x_shortcut_weight = \
            torch.tanh(self.bn_forget(torch.flatten(self.fc1(x_shortcut_weight), start_dim=1))) \
                .reshape(-1, C // self.forget_factor, 1, 1)
        x_shortcut_weight = torch.sigmoid(self.fc2(x_shortcut_weight))

        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        context_weight_3x3 = \
            F.softmax(torch.flatten(self.global_context3x3(x_3x3), start_dim=1), 1).reshape(-1, 1, H, W)
        context_weight_5x5 = \
            F.softmax(torch.flatten(self.global_context5x5(x_5x5), start_dim=1), 1).reshape(-1, 1, H, W)
        x_3x3 = self.global_pool(x_3x3 * context_weight_3x3) * H * W
        x_5x5 = self.global_pool(x_5x5 * context_weight_5x5) * H * W
        x_concat = torch.cat([x_3x3, x_5x5], 1)

        attention = torch.tanh(self.bn_attention(torch.flatten(self.fcall(x_concat), start_dim=1))) \
            .reshape(-1, 2 * C // self.reduction_factor, 1, 1)
        weight_3x3 = torch.unsqueeze(torch.flatten(self.fc3x3(attention), start_dim=1), 1)
        weight_5x5 = torch.unsqueeze(torch.flatten(self.fc5x5(attention), start_dim=1), 1)
        weight_all = F.softmax(torch.cat([weight_3x3, weight_5x5], 1), 1)
        weight_3x3, weight_5x5 = weight_all[:, 0, :].reshape(-1, C, 1, 1), weight_all[:, 1, :].reshape(-1, C, 1, 1)
        new_x = weight_3x3 * x_3x3 + weight_5x5  * x_5x5
        x = x_shortcut * x_shortcut_weight + new_x

        return torch.cat([x_dense, x], 1)

def _HybridBlock(num_layers, in_channels, bn_size, growth_rate):
    layers = []
    for i in range(num_layers):
        layers.append(_SMG(in_channels + growth_rate * i, growth_rate, bn_size))
    return nn.Sequential(*layers)

class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels, stride_3x3,
                 forget_factor=4, reduction_factor=4,):
        super(_Transition, self).__init__()
        self.in_channels = in_channels
        self.forget_factor = forget_factor
        self.reduction_factor = reduction_factor
        self.out_channels = out_channels
        self.reduce_channels = (in_channels - out_channels) // 2
        self.conv1_1x1 = BasicConv(in_channels, in_channels-self.reduce_channels, kernel_size=1, stride=1)
        self.conv2_3x3 = BasicConv(in_channels-self.reduce_channels, out_channels, kernel_size=3, stride=stride_3x3,
                                   padding=1, groups=1)
        # Mobile
        self.conv_3x3 = BasicConv(out_channels, out_channels, kernel_size=3,
                                  stride=1, padding=1, groups=out_channels)
        self.conv_5x5 = BasicConv(out_channels, out_channels, kernel_size=3,
                                  stride=1, padding=2, dilation=2, groups=out_channels)

        # GTSK layers
        self.global_context3x3 = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.global_context5x5 = nn.Conv2d(out_channels, 1, kernel_size=1)

        self.fcall = build_conv_layer(None, 2 * out_channels, 2 * out_channels // self.reduction_factor, kernel_size=1)
        self.bn_attention = nn.BatchNorm1d(2 * out_channels // self.reduction_factor)
        self.fc3x3 = build_conv_layer(None, 2 * out_channels // self.reduction_factor, out_channels, kernel_size=1)
        self.fc5x5 = build_conv_layer(None, 2 * out_channels // self.reduction_factor, out_channels, kernel_size=1)

        # SE layers
        self.global_forget_context = build_conv_layer(None, out_channels, 1, kernel_size=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn_forget = nn.BatchNorm1d(out_channels // self.forget_factor)
        self.fc1 = build_conv_layer(None, out_channels, out_channels // self.forget_factor, kernel_size=1)
        self.fc2 = build_conv_layer(None, out_channels // self.forget_factor, out_channels, kernel_size=1)

    def forward(self, x):
        post_x, x = self.conv1_1x1(x, is_return=True)
        x = self.conv2_3x3(x)

        H, W = x.size(2), x.size(3)
        C = x.size(1)
        x_shortcut = x
        forget_context_weight = self.global_forget_context(x_shortcut)
        forget_context_weight = torch.flatten(forget_context_weight, start_dim=1)
        forget_context_weight = F.softmax(forget_context_weight, 1)
        forget_context_weight = forget_context_weight.reshape(-1, 1, H, W)
        x_shortcut_weight = self.global_pool(x_shortcut * forget_context_weight) * H * W
        x_shortcut_weight = \
            torch.tanh(self.bn_forget(torch.flatten(self.fc1(x_shortcut_weight), start_dim=1))) \
                .reshape(-1, C // self.forget_factor, 1, 1)
        x_shortcut_weight = torch.sigmoid(self.fc2(x_shortcut_weight))

        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        context_weight_3x3 = \
            F.softmax(torch.flatten(self.global_context3x3(x_3x3), start_dim=1), 1).reshape(-1, 1, H, W)
        context_weight_5x5 = \
            F.softmax(torch.flatten(self.global_context5x5(x_5x5), start_dim=1), 1).reshape(-1, 1, H, W)
        x_3x3 = self.global_pool(x_3x3 * context_weight_3x3) * H * W
        x_5x5 = self.global_pool(x_5x5 * context_weight_5x5) * H * W
        x_concat = torch.cat([x_3x3, x_5x5], 1)
        attention = torch.tanh(self.bn_attention(torch.flatten(self.fcall(x_concat), start_dim=1))) \
            .reshape(-1, 2 * C // self.reduction_factor, 1, 1)
        weight_3x3 = torch.unsqueeze(torch.flatten(self.fc3x3(attention), start_dim=1), 1)
        weight_5x5 = torch.unsqueeze(torch.flatten(self.fc5x5(attention), start_dim=1), 1)
        weight_all = F.softmax(torch.cat([weight_3x3, weight_5x5], 1), 1)
        weight_3x3, weight_5x5 = weight_all[:, 0, :].reshape(-1, C, 1, 1), weight_all[:, 1, :].reshape(-1, C, 1, 1)
        new_x = weight_3x3 * x_3x3 + weight_5x5 * x_5x5
        x = x_shortcut * x_shortcut_weight + new_x
        return post_x, x

class post_BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, dilation=1, conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(post_BasicConv, self).__init__()

        self.conv = build_conv_layer(
            conv_cfg,
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=False)
        self.norm_name, norm = build_norm_layer(norm_cfg, out_channels, postfix=1)
        self.add_module(self.norm_name, norm)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


@BACKBONES.register_module
class HCGNet(nn.Module):
    def __init__(self, growth_rate=(32, 48, 64, 96), block_config=(3, 6, 12, 8),
                 bn_size=4, out_indices=(1, 2, 3, 4),
                 frozen_stages=1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True):
        super(HCGNet, self).__init__()

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        self.stem0 = post_BasicConv(3, 32, kernel_size=3, stride=2, padding=1)
        self.stem1 = post_BasicConv(32, 32, kernel_size=3, padding=1)
        self.stem2 = post_BasicConv(32, 64, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_feature = 64

        self.hybridblock1 = _HybridBlock(block_config[0], num_feature, bn_size, growth_rate[0])
        num_feature = num_feature + growth_rate[0] * block_config[0]
        outf = int(num_feature * 0.5)

        self.transition2 = _Transition(num_feature, outf, stride_3x3=2)
        num_feature = outf
        self.hybridblock2 = _HybridBlock(block_config[1], num_feature, bn_size, growth_rate[1])
        num_feature = num_feature + growth_rate[1] * block_config[1]
        outf = int(num_feature * 0.5)

        self.transition3 = _Transition(num_feature, outf, stride_3x3=2)
        num_feature = outf
        self.hybridblock3 = _HybridBlock(block_config[2], num_feature, bn_size, growth_rate[2])
        num_feature = num_feature + growth_rate[2] * block_config[2]
        outf = int(num_feature * 0.5)

        self.transition4 = _Transition(num_feature, outf, stride_3x3=2)
        num_feature = outf
        self.hybridblock4 = _HybridBlock(block_config[3], num_feature, bn_size, growth_rate[3])
        num_feature = num_feature + growth_rate[3] * block_config[3]

        self.norm_final_name, norm_final = build_norm_layer(norm_cfg, num_feature, postfix=1)
        self.add_module(self.norm_final_name, norm_final)
        self._freeze_stages()

        self.feat_dim = num_feature

    @property
    def norm_final(self):
        return getattr(self, self.norm_final_name)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem0.norm.eval()
            self.stem1.norm.eval()
            self.stem2.norm.eval()
            for m in [self.stem0, self.stem1, self.stem2]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'hybridblock{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.stem0(x)
        x = self.stem1(x)
        x = self.stem2(x)
        x = self.max_pool(x)
        outs = []
        x = self.hybridblock1(x)
        post_x, x = self.transition2(x)
        outs.append(post_x)
        x = self.hybridblock2(x)
        post_x, x = self.transition3(x)
        outs.append(post_x)
        x = self.hybridblock3(x)
        post_x, x = self.transition4(x)
        outs.append(post_x)
        x = self.hybridblock4(x)
        outs.append(F.relu(self.norm_final(x)))
        return tuple(outs)

    def init_weights(self, pretrained=None):
        '''
        with open('re.txt', 'a+') as f:
            for m in self.modules():
                print(m.parameter)
                #f.write(str(m) + '\n')
                exit()
        '''

        if isinstance(pretrained, str):
            checkpoint = torch.load(pretrained)
            param_dict = {}
            for k, v in zip(self.state_dict().keys(), checkpoint['state_dict'].keys()):
                param_dict[k] = checkpoint['state_dict'][v]
            self.load_state_dict(param_dict)
            #logger = logging.getLogger()
            #load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def train(self, mode=True):
        super(HCGNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
