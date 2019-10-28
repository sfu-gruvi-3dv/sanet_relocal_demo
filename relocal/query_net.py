import torch
import torch.nn as nn
import torch.nn.functional as F
import relocal.backbone_drn as drn
from core_dl.base_net import BaseNet
from relocal.unet_base import *
import shutil
import os
from relocal.pointnet2.pointnet2_utils import pytorch_utils as pt_utils

class UNet_4levels(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_4levels, self).__init__()
        # self.inc = inconv(in_channels, 512)
        self.down1 = down_cov_stride2(in_channels, 256)
        self.down2 = down_cov_stride2(256, 256)
        self.down3 = down_cov_stride2(256, 256)
        self.down4 = down_cov_stride2(256, 256)
        self.corres_query = nn.AvgPool2d(2)
        #     nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=2, stride=1, bias=False),
        #     nn.ReLU(inplace=True)
        # )
        self.up1 = up(512 + 256 + 256, 256)
        self.up2 = up(256 + 256, 256)
        # self.up3 = up(512 + 128, 64)
        # self.up4 = up(512 + 64, 64)
        self.outc = outconv(256, out_channels)
        self.out_logvar = nn.Conv2d(256, 1, kernel_size=1)
        self.corres_scene = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, scene_feat):
        # x1 = self.inc(x)        # out_c: 256
        x2 = self.down1(x)      # out_c: 256
        x3 = self.down2(x2)     # out_c: 256
        x4 = self.down3(x3)     # out_c: 256
        x5 = self.down4(x4)     # out_c: 256
        query_corres_vec = self.corres_query(x5)    # (N, 256, 1, 1)
        N, q_C, _, _ = query_corres_vec.shape
        query_corres_vec = query_corres_vec.view(N, 1, q_C)

        N, L, s_C = scene_feat.shape
        scene_corres_vec = self.corres_scene(scene_feat.view(N*L, s_C, 1, 1)).view(N, L, q_C)    # (N, L, C)
        dot_feat = query_corres_vec * scene_corres_vec
        corres_features = torch.sum(dot_feat, dim=-1, keepdim=True)    # (N, L, 1)
        weights = F.softmax(corres_features, dim=1)    # (N, L, 1)
        scene_feat = torch.sum(scene_feat * weights, dim=1)    # (N, C)

        x5 = torch.cat([x5, scene_feat.view(N, s_C, 1, 1).expand(N, s_C, x5.shape[2], x5.shape[3])], dim=1)
        x6 = self.up1(x5, x4)    # in_c: 512 + 256 + 256 -> 256
        x7 = self.up2(x6, x3)     # in_c: 256 + 256 -> 256
        # x8 = self.up3(x7, x2)     # in_c: 256 + 128 -> 64
        # x_up4 = self.up4(x8, x)     # in_c: 512 + 64 -> 64
        out = self.outc(x7)
        out_logvar = self.out_logvar(x7)
        return out, out_logvar, x2, x3, x7


class UNet_2levels(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_2levels, self).__init__()
        self.inc = inconv(in_channels, 256)
        self.down1 = down(256, 512)
        self.down2 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 256)
        self.outc = outconv(256, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        return x


BatchNorm = nn.BatchNorm2d


class QueryNet(BaseNet):
    def __init__(self, input_dim=[(3, 256, 256), (512,)]):
        """
        construct QueryNet, input an image & a scene global feature, output scene coordinate of the image
        :param feat_dim: size of the scene global feature
        """
        super(QueryNet, self).__init__()
        self.input_shape_chw = input_dim
        feat_dim = input_dim[1][0]

        drn_module = drn.drn_d_54(pretrained=True)  # use DRN54 for now
        self.pre_base_model = nn.Sequential(
            drn_module.layer0,
            drn_module.layer1,
            drn_module.layer2,
            drn_module.layer3
        )
        self.base_model = nn.Sequential(
            drn_module.layer4,
            nn.Conv2d(512, 256, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.down1 = down_cov_stride2(256, 256)
        self.down2 = down_cov_stride2(256, 256)
        self.down3 = down_cov_stride2(256, 256)
        self.down4 = down_cov_stride2(256, 256)
        self.up1 = up(512 + 256 + 256, 256)
        self.up2 = up(256 + 256, 256)
        # self.up3 = up(512 + 128, 64)
        # self.up4 = up(512 + 64, 64)
        self.outc = outconv(256, 3)
        self.out_logvar = nn.Conv2d(256, 1, kernel_size=1)
        self.corres_scene = pt_utils.SharedMLP([512, 256], bn=True)
        # self.inplanes = 256
        # self.unet = UNet_4levels(self.inplanes, 3)      # U-Net
        # self.layer5 = nn.Sequential(
        #     self._make_layer(drn.Bottleneck, 256, 6, dilation=2, new_level=False),
        #     drn_module.layer6,
        #     drn_module.layer7,
        #     drn_module.layer8,
        # )
        #
        # self.final_conv = nn.Sequential(
        #     nn.Conv2d(512, 3, kernel_size=1, stride=1, bias=False),
        #     # BatchNorm(256),
        #     # nn.ReLU(inplace=True),
        #     # nn.Conv2d(256, 3, kernel_size=1, stride=1, bias=False),
        # )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
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

    def forward(self, x):
        """
        forward with image & scene feature
        :param image: (N, C, H, W)
        :param scene_feat: (N, L, 512, 4, 4)
        :return:
        """
        image, scene_feat = x

        # Feature Extract ----------------------------------------------------------------------------------------------
        # N, C1 = feat.shape
        query_img_feat3 = self.pre_base_model(image)          #(N, 256, 64, 64)
        query_img_feat2 = self.base_model(query_img_feat3)    #(N, 512, 32, 32)
        # f = torch.cat([I_f, feat.view(N, C1, 1, 1).expand(N, C1, I_f.shape[2], I_f.shape[3])], dim=1)
        # out = self.final_conv(self.layer5(f))

        # UNet Down ----------------------------------------------------------------------------------------------------
        x2 = self.down1(query_img_feat2)                      # (N, 256, 16, 16)
        x3 = self.down2(x2)                                   # (N, 256, 8, 8)
        x4 = self.down3(x3)                                   # (N, 256, 4, 4)
        x5 = self.down4(x4)                                   # (N, 256, 2, 2)

        # UNet (scene_feat RefineLayer, get weighted scene_feat) -------------------------------------------------------
        N, L, _, _, _ = scene_feat.shape
        scene_feat = scene_feat.permute(0, 2, 1, 3, 4).contiguous().view(N, 512, 1, L*4*4)
        scene_corres_vec = self.corres_scene(scene_feat)      # (N, 256, 1, L*4*4)
        query_corres_vec = x5.view(N, 256, 4, 1)              # (N, 256, 4, 1)
        dot_feat = query_corres_vec * scene_corres_vec
        corres_features = torch.sum(dot_feat, dim=1, keepdim=True)  # (N, 1, 4, L*4*4)
        weights = F.softmax(corres_features, dim=-1)                # (N, 1, 4, L*4*4)
        scene_feat = torch.sum(scene_feat * weights, dim=-1)        # (N, 512, 4)

        # UNet Up ------------------------------------------------------------------------------------------------------
        x5 = torch.cat([x5, scene_feat.view(N, 512, 2, 2)], dim=1)  # (N, 512 + 256, 2, 2)
        x6 = self.up1(x5, x4)                                       # (N, 256, 4, 4)
        x7 = self.up2(x6, x3)                                       # (N, 256, 8, 8)
        out = self.outc(x7)
        out_logvar = self.out_logvar(x7)

        rough_res_feat = x7
        query_img_feat0 = x3
        query_img_feat1 = x2
        return out, out_logvar, query_img_feat3, query_img_feat2, query_img_feat1, query_img_feat0, rough_res_feat

if __name__ == '__main__':

    # from tensorboardX import SummaryWriter
    #
    # Test ground
    net = QueryNet().cuda()
    net.summary()

# with SummaryWriter(log_dir='/mnt/Tango/corres_logs/logs/net_graph') as w:
#     w.add_graph(net, input_to_model=[torch.rand(1, 3, 256, 256).cuda(), torch.rand(1, 512).cuda()])
