import torch
import torch.nn as nn
import numpy as np
from core_dl.base_net import BaseNet


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation[1], bias=False,
                               dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class FeatureBaseExtractor(BaseNet):

    def __init__(self, in_dim=(12, 240, 320)):
        super(FeatureBaseExtractor, self).__init__()
        self.input_shape_chw = in_dim
        self.wrap_module = None

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_dim[0], 32, kernel_size=7, stride=1, padding=3, bias=False),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.inplanes = 32
        self.layer1 = self._make_conv_layers(32, 1, stride=1)
        self.layer2 = self._make_conv_layers(64, 1, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 1, stride=2)
        self.extras = nn.Sequential(
            self._make_layer(BasicBlock, 128, 1, stride=2),
            self._make_layer(BasicBlock, 256, 2, stride=2),
            self._make_layer(BasicBlock, 256, 2, stride=2)
        )

        self.pre_x_2d = None

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(self.inplanes, channels, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, I, d, flow):
        N, C, H, W = I.shape
        # if self.pre_x_2d is None or self.pre_x_2d.shape[0] != N:
        #     self.pre_x_2d = ba_module.x_2d_coords_torch((L - 1) * N, H, W)
        #
        # # Wrap the image
        # I_wrap, d_wrap = wrap(I, d, rel_T, K, pre_cached_x_2d=self.pre_x_2d)
        in_tensor = torch.cat([I[:-1, :, :, :, :], I[1:, :, :, :, :], I_wrap,
                           d[:-1, :, :, :, :], d[1:, :, :, :, :], d_wrap], dim=2).requires_grad_()

        assert in_L == L-1
        assert in_N == N
        assert in_C == 12
        assert in_H == H
        assert in_W == W

        # Extract features
        conv0 = self.layer0(in_tensor.view((L-1)*N, in_C, in_H, in_W))
        conv1 = self.layer1(conv0)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        y = self.extras(conv3)
        flatten = y.view((L-1), N, -1)
#         print('Output dim:', flatten.shape)
        return flatten


# Test
# net = FeatureBaseExtractor(in_dim=(6, 240, 320))
# net.cuda()
# net.summary()