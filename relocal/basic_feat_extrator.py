import torch
import torch.nn as nn
import torch.nn.functional as F
import relocal.backbone_drn as drn
from core_dl.base_net import BaseNet
from relocal.unet_base import *
import shutil
import os
from relocal.pointnet2.pointnet2_utils import pytorch_utils as pt_utils


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


class ContextNormFCN(nn.Module):
    def __init__(self, n_channels=[256, 128]):
        super(ContextNormFCN, self).__init__()
        self.conv0 = nn.Conv2d(n_channels[0], n_channels[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.context_norm = nn.InstanceNorm1d(n_channels[1], affine=False, track_running_stats=False)
        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(n_channels[1]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        N, _, M, K = x.shape
        x = self.conv0(x)
        x = x.view(N, -1, K)
        x = self.context_norm(x)
        x = x.view(N, -1, M, K)
        x = self.bn_relu(x)
        return x


class ContextAttention(nn.Module):
    def __init__(self, n_channel, out_xyz=True, use_pre_xyz=True):
        super(ContextAttention, self).__init__()
        self.out_xyz = out_xyz
        self.use_pre_xyz = use_pre_xyz
#         self.fcn0 = ContextNormFCN(n_channels=[n_channel * 2, n_channel])
        self.fcn0 = nn.Sequential(
            nn.Conv2d(n_channel * 2, n_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(inplace=True)
        )
        self.fcn1 = nn.Sequential(
#             ContextNormFCN(n_channels=[n_channel + 3, n_channel]),
#             ContextNormFCN(n_channels=[n_channel, n_channel])
            nn.Conv2d(n_channel + 3, n_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channel, n_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(inplace=True)
        )
        self.skip = nn.Conv2d(n_channel + 3, n_channel, kernel_size=1, stride=1, padding=0, bias=False)
        if out_xyz:
            if use_pre_xyz:
                self.out_conv = nn.Conv1d(n_channel + 3, 3, kernel_size=1, stride=1, padding=0, bias=False)
            else:
                self.out_conv = nn.Conv1d(n_channel, 3, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, query_rgb_feat, scene_rgb_feat, scene_xyz, pre_xyz=None, mask=None):
        """
        attention modul
        :param query_rgb_feat: (N, C, Hq * Wq, 1)
        :param scene_rgb_feat: (N, C, 1, nsample) or (N, C, Hq * Wq, nsample)
        :param scene_xyz: (N, 3, 1, nsample) or (N, 3, Hq * Wq, nsample)
        :param pre_xyz: (N, 3, Hq * Wq)
        :param mask: (N, 1, 1, nsample) or (N, 1, Hq * Wq, nsample)
        :return: weights: (N, 1, Hq * Wq, nsample)
        """
        N, C, M, _ = query_rgb_feat.shape
        _, _, _, K = scene_rgb_feat.shape
        if scene_rgb_feat.size(2) == 1:
            scene_rgb_feat = scene_rgb_feat.expand(-1, -1, M, -1)
        if scene_xyz.size(2) == 1:
            scene_xyz = scene_xyz.expand(-1, -1, M, -1)
        if mask.size(2) == 1:
            mask = mask.expand(-1, -1, M, -1)
        scene_xyz = scene_xyz * mask
        query_rgb_feat = query_rgb_feat.expand(-1, -1, -1, K)
        cat_feat = torch.cat([scene_rgb_feat, query_rgb_feat], dim=1)       # (N, 2*C, M, K)
        feat = self.fcn0(cat_feat)                                          # (N, C, M, K)
        cat_feat = torch.cat([feat, scene_xyz], dim=1)                      # (N, C+3, M, K)
        feat = self.fcn1(cat_feat)                                          # (N, C, M, K)
        skip_feat = self.skip(cat_feat)                                     # (N, C, M, K)
        feat = feat + skip_feat
        feat = F.max_pool2d(feat, kernel_size=(1, K)).squeeze(-1)           # (N, C, M)
        if self.use_pre_xyz:
            feat = torch.cat([feat, pre_xyz], dim=1)                            # (N, C+3, M)
        if self.out_xyz:
            out = self.out_conv(feat)                                           # (N, 3, M)
            sumed_mask = torch.sum(mask, dim=-1)                                # should be (N, 1, M)
            out = torch.where(torch.gt(sumed_mask, 1e-2).expand(-1, 3, -1), out, out)    # dim: (N, 3, M)
        else:
            out = feat
            sumed_mask = torch.sum(mask, dim=-1)                                # should be (N, 1, M)
            out = torch.where(torch.gt(sumed_mask, 1e-2).expand(-1, out.size(1), -1), out, out)  # dim: (N, 3, M)
        return out, None


class RGBNet(nn.Module):
    def __init__(self, input_dim=(3, 256, 256)):
        """
        Network for extracting RGB features
        :param input_dim: size of the input image
        """
        super(RGBNet, self).__init__()
        self.input_shape_chw = input_dim

        drn_module = drn.drn_d_38(pretrained=True)  # use DRN38 for now
        self.block0 = nn.Sequential(
            drn_module.layer0,
            drn_module.layer1
        )
        self.block1 = drn_module.layer2
        self.block2 = drn_module.layer3
        self.block3 = drn_module.layer4
        self.block4 = drn_module.layer5
        self.block5 = drn_module.layer6
        self.block6 = nn.Sequential(
            drn_module.layer7,
            drn_module.layer8
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        forward with image & scene feature
        :param image: (N, C, H, W)
        :return:
        """
        x0 = self.block0(x)     # 192x256
        x1 = self.block1(x0)    # 96x128
        x2 = self.block2(x1)    # 48x64
        x3 = self.block3(x2)    # 24x32
        x4 = self.block4(x3)    # 12x16
        x5 = self.block5(x4)    # 6x8
        x6 = self.block6(x5)    # 3x4
        x7 = self.block7(x6)    # 2x2
        return x7, x6, x5, x4, x3, x2, x1, x0


class CoordNet(nn.Module):
    def __init__(self, input_dim=(3, 256, 256)):
        """
        Network for extracting RGB features
        :param input_dim: size of the input image
        """
        super(CoordNet, self).__init__()
        self.input_shape_chw = input_dim

        drn_module = drn.drn_d_38(pretrained=True)  # use DRN38 for now
        self.block0 = nn.Sequential(
            drn_module.layer0,
            drn_module.layer1
        )
        self.block1 = drn_module.layer2
        self.block2 = drn_module.layer3
        self.block3 = drn_module.layer4
        self.block4 = drn_module.layer5
        self.block5 = drn_module.layer6
        self.block6 = nn.Sequential(
            drn_module.layer7,
            drn_module.layer8
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        forward with image & scene feature
        :param image: (N, C, H, W)
        :return:
        """
        x0 = self.block0(x)     # 192x256
        x1 = self.block1(x0)    # 96x128
        x2 = self.block2(x1)    # 48x64
        x3 = self.block3(x2)    # 24x32
        x4 = self.block4(x3)    # 12x16
        x5 = self.block5(x4)    # 6x8
        x6 = self.block6(x5)    # 3x4
        x7 = self.block7(x6)    # 2x2
        return x7, x6, x5, x4, x3, x2, x1, x0

if __name__ == '__main__':
    from core_dl.module_util import summary_layers
    model = CoordNet().cuda()
    summary_layers(model, input_size=(3, 192, 256))
