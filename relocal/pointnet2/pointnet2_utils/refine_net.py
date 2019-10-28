import torch
import torch.nn as nn
import torch.nn.functional as F

from . import pointnet2_utils
from . import pytorch_utils as pt_utils
from typing import List, Tuple
from relocal.basic_feat_extrator import ContextAttention


BatchNorm = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, padding=0, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
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


class BasicBlock1x1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock1x1, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride,
                             padding=0, dilation=dilation[0])
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes,
                             padding=0, dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
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


class RefineLayer(nn.Module):
    r"""Select scene features basing on previous coordinate prediction and do refinement

    Parameters
    ----------
    radius : float32
        radius to group with
    nsample : int32
        Number of samples in each ball query
    use_xyz: bool
        Whether use xyz coordinate as feature
    nblocks : int
        Number of ResBlocks for this refine layer
    inplanes_res: int,
        Number of input channels for ResBlocks
    planes: int,
        Number of output channels
    """

    def __init__(
            self,
            radius: float,
            nsample: int,
            use_xyz: bool = True,
            rgb_planes: int = 256,
            pre_planes: int = 256,
            is_final: bool = False
    ):
        super().__init__()
        self.nsample = nsample
        self.is_final = is_final
        self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
        self.attention = ContextAttention(rgb_planes, out_xyz=is_final, use_pre_xyz=True)

        if not is_final:
            skip = nn.Sequential(
                nn.Conv2d(rgb_planes + pre_planes, rgb_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(rgb_planes)
            )
            self.fusion = nn.Sequential(
                BasicBlock(rgb_planes + pre_planes, rgb_planes, downsample=skip),
            )
            
            skip = nn.Sequential(
                nn.Conv2d(rgb_planes + rgb_planes + 3, rgb_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(rgb_planes)
            )
            self.res_blocks = nn.Sequential(
                # nn.Conv2d(rgb_planes + rgb_planes + 3, rgb_planes, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(rgb_planes),
                # nn.ReLU(inplace=True),
                BasicBlock(rgb_planes + rgb_planes + 3, rgb_planes, downsample=skip)
            )
            
            self.out_conv = nn.Conv2d(rgb_planes, 3, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor,
                features: torch.Tensor = None, query_img_feat: torch.Tensor = None, pre_feat: torch.Tensor = None,
                query_X_world: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, H*W, 3) tensor of the new features' xyz (centriods)
        features : torch.Tensor
            (B, C*2, N) tensor of the descriptors of the the features
        query_img_feat : torch.Tensor
            (B, C, H, W) tensor of the query rgb features
        query_X_world : torch.Tensor
            (B, 3, H, W) tensor of the ground truth output coordinate

        Returns
        -------
        """

        B, C, H, W = query_img_feat.shape
        # select scene features near the centroids by ball query -------------------------------------------------------
        new_features = self.grouper(
            xyz, new_xyz, features
        )  # (B, C*2, H*W, nsample)

        corres_features = new_features[:, 3:3 + C, :, :]                            # (B, C, H*W, nsample)
        geo_features = new_features[:, :3, :, :]                                    # (B, 3, H*W, nsample)
        mask = new_features[:, -1:, :, :]                                           # (B, 1, H*W, nsample)
        pre_xyz = new_xyz.permute(0, 2, 1).contiguous()                             # (B, 3, H*W)

        # compute gathered features for each query pixel ---------------------------------------------------------------
        sumed_geo_features, confid = self.attention(query_img_feat.view(B, C, H * W, 1), corres_features, geo_features,
                                                    pre_xyz=pre_xyz, mask=mask)

        if query_X_world is not None:
            diff = query_X_world.view(B, 3, H * W, 1) - geo_features                    # (B, 3, H*W, nsample)
            dist = torch.norm(diff, p=2, dim=1, keepdim=True)                           # (B, 1, H*W, nsample)
            dist_topk, dist_idx_topk = torch.topk(dist, 8, dim=-1, largest=False)       # (B, 1, H*W, K)
            weights_all_gt = torch.zeros_like(dist).scatter_(-1, dist_idx_topk, torch.ones_like(dist_topk))  # (B, 1, H*W, nsample)
            weights_all = confid                                                        # (B, 1, H*W, nsample)
            weights_all_gt = weights_all_gt.squeeze(1)                                  # (B, H*W, nsample)
            # weights_all = weights_all.squeeze(1)                                      # (B, H*W, nsample)
        else:
            weights_all_gt = None
            weights_all = None

        # compute refined xyz ------------------------------------------------------------------------------------------
        if self.is_final:
            refined_xyz = sumed_geo_features.view(B, 3, H, W)                           # (B, 3, H, W)
            return refined_xyz, refined_xyz, weights_all, weights_all_gt
        else:
            fused_feat = torch.cat([query_img_feat, pre_feat], dim=1)               # (B, C+pre_C, H, W)
            fused_feat = self.fusion(fused_feat)                             # (B, C, H, W)
            sumed_geo_features = sumed_geo_features.view(B, C+3, H, W)              # (B, C+3, H, W)
            feat = torch.cat([fused_feat, sumed_geo_features], dim=1)               # (B, C+C+3, H, W)
            feat = self.res_blocks(feat)                                  # (B, C, H, W)
            refined_xyz = self.out_conv(feat)                               # (B, 3, H, W)
            return refined_xyz, feat, weights_all, weights_all_gt
