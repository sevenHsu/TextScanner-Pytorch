# /usr/bin/python
# -*- coding:utf-8 -*-
"""
    @description: TextScanner base mode
    @detail:
    @copyright: Chonqqing Ainnovation Tech Co., Ltd.
    @author: Seven Hsu
    @e-mail: xushen@ainnovation.com
    @date: 2020-12-21
"""
import torch
import torch.nn as nn
from torch.utils import model_zoo

__all__ = ['txt_scan_res18', 'txt_scan_res34']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, attn=False):
        super(ResNet, self).__init__()
        self.attn = attn
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=2)
        self.attn1 = nn.Sequential(nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
                                   nn.Softmax2d())

        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.attn2 = nn.Sequential(nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
                                   nn.Softmax2d())

        self.layer3 = self._make_layer(block, 512, layers[2], stride=1)
        self.attn3 = nn.Sequential(nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
                                   nn.Softmax2d())

        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.attn4 = nn.Sequential(nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
                                   nn.Softmax2d())

        self.layer5 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))

        self.up_layer1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.up_layer2 = nn.Conv2d(640, 128, kernel_size=3, stride=1, padding=1)

        self.up_layer3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                       nn.Conv2d(384, 128, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        self.up_layer4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                       nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))  # ,

        self.out_layer = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        map1 = self.layer1(x)
        map2 = self.layer2(map1)
        map3 = self.layer3(map2)
        map4 = self.layer4(map3)
        attn4 = self.attn4(map4)
        map5 = self.layer5(map4)
        if self.attn:
            attn1 = self.attn1(map1)
            attn2 = self.attn2(map2)
            attn3 = self.attn3(map3)

        if self.attn:
            map3 = map3 * (attn3[:, 1:2, :, :] + 1)
            map2 = map2 * (attn2[:, 1:2, :, :] + 1)
            map1 = map1 * (attn1[:, 1:2, :, :] + 1)
        upmap5 = self.relu(map5 * (attn4[:, 1:2, :, :] + 1))
        upmap4 = self.relu(self.up_layer1(upmap5))
        upmap3 = self.relu(self.up_layer2(torch.cat((upmap4, map3), dim=1)))
        upmap2 = self.up_layer3(torch.cat((upmap3, map2), dim=1))
        upmap1 = self.up_layer4(torch.cat((upmap2, map1), dim=1))
        output = self.out_layer(upmap1)

        return output

    def forward(self, x):
        return self._forward_impl(x)


model_blocks = {
    'resnet18': BasicBlock,
    'resnet34': BasicBlock,
    'resnet50': Bottleneck
}
model_layers = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3]
}


# order segmentation branch
class OrderSeg(nn.Module):
    def __init__(self, input_size=(48, 192), max_seq=16):
        super(OrderSeg, self).__init__()
        self.input_size = input_size
        self.max_seq = max_seq

        self.down1 = conv3x3(128, 16, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.down2 = conv3x3(16, 32, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.down3 = conv3x3(32, 64, 2)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # input_size: h*c,hidden_size: h*c
        self.rnn = nn.GRU(input_size=self.input_size[0] * 64 // 8, hidden_size=self.input_size[0] * 64 // 8,
                          batch_first=True)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, self.max_seq, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        down1_map = self.down1(x)
        down1_map = self.relu(self.bn1(down1_map))

        down2_map = self.down2(down1_map)
        down2_map = self.relu(self.bn2(down2_map))

        down3_map = self.down3(down2_map)
        down3_map = self.relu(self.bn3(down3_map))

        # N,C,H,W->N,W,C,H
        rnn_input = down3_map.permute([0, 3, 1, 2])
        # N,W,C,H->N,W,C*H
        rnn_input = rnn_input.reshape([down3_map.shape[0], down3_map.shape[3], -1])

        # h0 = torch.rand(1, down3_map.shape[0], rnn_input.shape[-1]).cuda(self.gpu)
        out, hid = self.rnn(rnn_input)  # ,h0)
        # N,W,C*H->N,W,C,H
        rnn_output = out.reshape([out.shape[0], out.shape[1], -1, down3_map.shape[2]])
        # N,W,C,H->N,C,H,W
        rnn_output = rnn_output.permute([0, 2, 3, 1])

        up1_map = self.up3(rnn_output)
        up2_map = self.up2(torch.add(up1_map, down2_map))
        up3_map = self.up1(torch.add(up2_map, down1_map))

        return up3_map


# character segmentation bracnch
def char_seg(num_classes):
    return nn.Sequential(conv3x3(128, 64, 1), conv1x1(64, num_classes, 1))


# location segmentation branch
def pos_seg():
    return nn.Sequential(conv3x3(128, 64, 1), conv1x1(64, 1, 1), nn.Sigmoid())


def load_weight(model, weight):
    """
    load pre-trained on ImageNet weight
    :param model:
    :param weight:
    :return:
    """
    model_dict = model.state_dict()
    weight = {k: v for k, v in weight.items()
              if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(weight)
    model.load_state_dict(model_dict)


class TxtScanNet(nn.Module):
    def __init__(self, basenet='resnet18', input_size=(96, 384), max_seq=16, num_class=63, mode='test', attn=False):
        super(TxtScanNet, self).__init__()
        self.attn = attn
        block = model_blocks[basenet]
        layers = model_layers[basenet]

        self.mode = mode
        # N (max length of the text sequence)
        self.max_seq = max_seq
        # C (character num_classes)
        self.num_classes = num_class
        # H,W (size of input image)
        self.input_size = input_size

        # build network structure
        self.backbone = ResNet(block, layers, attn=self.attn)
        self.charseg = char_seg(self.num_classes)
        self.posseg = pos_seg()
        self.ordseg = OrderSeg(input_size=(self.input_size[0] // 2, self.input_size[1] // 2), max_seq=self.max_seq)

        # load pre_trained weight and initialize module weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for weight in m._flat_weights[:2]:
                    nn.init.xavier_normal_(weight)
        if self.mode == 'train':
            load_weight(self.backbone, model_zoo.load_url(model_urls[basenet]))

    def forward(self, x):
        # B,128,h,w (h=H/2,w=W/2)
        x = self.backbone(x)
        # B,C,h,w
        chars_seg_map = self.charseg(x)
        # B,1,h,w
        pos_seg_map = self.posseg(x)
        # B,N,h,w
        ord_seg_map = self.ordseg(x)

        return chars_seg_map, ord_seg_map, pos_seg_map


def txt_scan_res18(basenet, input_size, max_seq, num_class, mode, attn):
    net = TxtScanNet('resnet18', input_size, max_seq, num_class, mode, attn)
    return net


def txt_scan_res34(basenet, input_size, max_seq, num_class, mode, attn):
    net = TxtScanNet('resnet34', input_size, max_seq, num_class, mode, attn)
    return net
