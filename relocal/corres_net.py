import numpy as np
import torch
import torch.nn as nn
import relocal.corres_net_backbone as drn
from relocal.query_net import QueryNet
from core_dl.base_net import BaseNet
import inspect
import shutil
import os
from relocal.unet_base import *
from relocal.point_feat import *
from relocal.pointnet2.pointnet2_utils.refine_net import RefineLayer
from relocal.basic_feat_extrator import RGBNet, CoordNet, ContextAttention
from relocal.pointnet2.pointnet2_utils.refine_net import BasicBlock


''' Local feature refinements ------------------------------------------------------------------------------------------
'''
class Corres2D3DNet(BaseNet):

    def __init__(self, out_global_feat=512):
        super(Corres2D3DNet, self).__init__()
        self.input_shape_chw = (6, 192, 256)
        # drn_module = drn.drn_rgbxyz_d_54(pretrained=True, num_feat=out_global_feat)
        # self.pre_base_model = nn.Sequential(
        #     drn_module.layer0,
        #     drn_module.layer1,
        #     drn_module.layer2,
        #     drn_module.layer3
        # )
        # self.base_model = nn.Sequential(
        #     drn_module.layer4,
        #     drn_module.layer5,
        #     drn_module.layer6,
        #     drn_module.layer7,
        #     # drn_module.layer8
        # )
        # self.scene_feat_model = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        # )
        # # self.avgpool = nn.AvgPool2d(4)
        # # self.fc = nn.Conv2d(512, out_global_feat, kernel_size=(8, 8), stride=1, padding=0, bias=True)
        # self.query_net = QueryNet(input_dim=[(3, 256, 256), (512, )])
        # self.refine_net0 = RefineLayer((8, 8), 1.0, 64, False, 2, 512, 256, 256, 256)
        # self.refine_net1 = RefineLayer((16, 16), 0.5, 16, False, 2, 512, 256, 256, 256)
        # self.refine_net2 = RefineLayer((32, 32), 0.25, 16, False, 2, 256, 256, 256, 256)

        self.rgb_net = RGBNet(input_dim=(3, 192, 256))
#         self.coord_net = CoordNet(input_dim=(3, 192, 256))
#         self.up0 = up(1024 + 512, 512, out_size=(3, 4))
#         self.up1 = up(1024, 512)
#         self.rough_outconv = nn.Sequential(
#             nn.Conv2d(512 + 512, 512, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 3, kernel_size=1, stride=1, padding=0, bias=False)
#         )
        skip = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512)
        )
        self.res_blocks = nn.Sequential(
            BasicBlock(512 + 512, 512, downsample=skip)
        )
        self.rough_outconv = nn.Conv2d(512, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.refine_net0 = RefineLayer(radius=0.75, nsample=64, use_xyz=True, rgb_planes=512, pre_planes=512, is_final=False)
        self.refine_net1 = RefineLayer(radius=0.5, nsample=64, use_xyz=True, rgb_planes=256, pre_planes=512, is_final=False)
        self.refine_net2 = RefineLayer(radius=0.25, nsample=64, use_xyz=True, rgb_planes=128, pre_planes=256, is_final=False)
        self.refine_net3 = RefineLayer(radius=0.125, nsample=64, use_xyz=True, rgb_planes=64, pre_planes=128, is_final=True)
        self.attention = ContextAttention(512, out_xyz=False, use_pre_xyz=False)

    def build_point_cloud_feat(self, scene_coord, valid_mask, scene_rgb_coord_feat, shuffle=True):
        """
        :param scene_coord: dim (N, L, 3, H, W)
        :param valid_mask: dim (N, L, 3, H, W)
        :param scene_rgb_coord_feat: (N, L, C, H, W), where C = [c_rgb, c_coords]
        :return: point positions and feature
        """

        N, L, _, H, W = scene_coord.shape
        C = scene_rgb_coord_feat.size(2)

        feat_H, feat_W = scene_rgb_coord_feat.size(3), scene_rgb_coord_feat.size(4)
        pool_scale_ratio = H // feat_H
        pt_pos, pt_feat, pt_mask = extract_points_validpool(scene_coord.detach().view(N*L, -1, H, W),
                                                            valid_mask.detach().view(N*L, -1, H, W),
                                                            scene_rgb_coord_feat.view(N*L, -1, feat_H, feat_W),
                                                            pool_kernel_size=pool_scale_ratio,
                                                            point_chw_order=True)

        if not shuffle:
            pt_pos = pt_pos.view(N, L, 3, feat_H, feat_W)
            pt_feat = torch.cat([pt_feat.view(N, L, C, feat_H, feat_W), pt_mask.view(N, L, 1, feat_H, feat_W).float()], dim=2)
            return pt_pos, pt_feat
            
        pt_pos = pt_pos.view(N, L, 3, feat_H * feat_W).permute(1, 3, 0, 2).\
            contiguous().view(L * feat_H * feat_W, N, 3)                                        # dim: (L*h*w, N, 3)
        pt_feat = pt_feat.view(N, L, C, feat_H * feat_W).permute(1, 3, 0, 2).\
            contiguous().view(L * feat_H * feat_W, N, C)                                        # dim: (L*h*w, N, C)
        pt_mask = pt_mask.view(N, L, feat_H * feat_W).permute(1, 2, 0). \
            contiguous().view(L * feat_H * feat_W, N, 1).float()                                # dim: (L*h*w, N, 1)

        # Random shuffle point cloud -----------------------------------------------------------------------------------
        pt_pos_feat = torch.cat([pt_pos, pt_feat, pt_mask], dim=-1)                             # dim: (L*h*w, N, 3+C+1)
        perm = torch.randperm(L * feat_H * feat_W)
        pt_pos_feat = pt_pos_feat[perm]                                                         # dim: (L*h*w, N, 3+C+1)
        pt_pos = pt_pos_feat[:, :, :3].permute(1, 0, 2).contiguous()                            # dim: (N, L*h*w, 3)
        pt_feat = pt_pos_feat[:, :, 3:].permute(1, 2, 0).contiguous()                           # dim: (N, C+1, L*h*w)

        # Add fake point -----------------------------------------------------------------------------------------------
        fake_point = torch.full((N, 1, 3), 999.0).to(pt_pos.device)
        pt_pos = torch.cat([fake_point, pt_pos], dim=1)

        fake_pt_feat = torch.zeros(N, pt_feat.size(1), 1).to(pt_feat.device)
        pt_feat = torch.cat([fake_pt_feat, pt_feat], dim=2)

        return pt_pos, pt_feat

    def forward(self, query_rgb, scene, scene_valid_mask, query_X_world=None):
        """
        :param query_rgb: dim: (3, 192, 256)
        :param scene: dim: (N, L, 6, 192, 256), first 3 channel is rgb, second 3 channel is coords
        :param scene_valid_mask: dim: (192, 256)
        :param query_X_world: dim: [(3, h, w)]
        """
        N, L, _, H, W = scene.shape

        scene_rgb = scene[:, :, :3, :, :]
        scene_coord = scene[:, :, 3:, :, :]

        # Compute RGB features for both scene and query  ---------------------------------------------------------------
        # concatenate together due to Batch Normalization
        query_rgb = query_rgb.view(N, 1, -1, H, W)
        scene_query_rgb = torch.cat((scene_rgb, query_rgb), dim=1)                         # dim: (N, L+1, 3, H, W)
        scene_query_rgb = scene_query_rgb.view(N*(L+1), -1, H, W)                          # dim: (N*(L+1), 3, H, W)
        scene_query_rgb_feats = self.rgb_net(scene_query_rgb)                              # dim: [(N*(L+1), C, h, w)]

        # extract scene and query rgb features
        scene_rgb_feats = [f.view(N, L+1, f.size(1), f.size(2), f.size(3))[:, :L, :, :, :]
                           for f in scene_query_rgb_feats]                                 # dim: [(N, L, C, h, w)]
        query_rgb_feats = [f.view(N, L+1, f.size(1), f.size(2), f.size(3))[:, L, :, :, :]
                           for f in scene_query_rgb_feats]                                 # dim: [(N, C, h, w)]

        # Compute Coords features for scene ----------------------------------------------------------------------------
#         scene_coord_feats = self.coord_net(scene_coord.view(N*L, -1, H, W))                # dim: [(N*L, C, h, w)]
#         scene_coord_feats = [f.view(N, L, f.size(1), f.size(2), f.size(3))
#                              for f in scene_coord_feats]                                   # dim: [(N, L, C, h, w)]

        # Build point clouds with feature ------------------------------------------------------------------------------
        # concatenate rgb & scene coord feature
#         scene_feats = [torch.cat((scene_rgb_feats[i], scene_coord_feats[i]), dim=2)        # dim:(N, L, c+c*, h, w)
#                                  for i in [2, 3, 4, 5]]
#         scene_feats = [
#             F.unfold(
#                 scene_rgb_feats[i].contiguous().view(N*L, scene_rgb_feats[i].size(2), scene_rgb_feats[i].size(3), scene_rgb_feats[i].size(4)),
#                 kernel_size=4,
#                 stride=4
#             ).view(N, L, scene_rgb_feats[i].size(2)*4*4, scene_rgb_feats[i].size(3) // 4, scene_rgb_feats[i].size(4) // 4)
#             for i in [4, 5, 6, 7]
#         ]        # dim:(N, L, c*x4x4, h, w)
#
#         query_feats = [
#             F.unfold(
#                 query_rgb_feats[i],
#                 kernel_size=4,
#                 stride=4
#             ).view(N, query_rgb_feats[i].size(1)*4*4, query_rgb_feats[i].size(2) // 4, query_rgb_feats[i].size(3) // 4)
#             for i in [4, 5, 6, 7]
#         ]        # dim:(N, L, c*x4x4, h, w)
        scene_feats = [scene_rgb_feats[i].contiguous() for i in [2, 3, 4, 5]]                            # dim:(N, L, c+c*, h, w)

        # compute scene local features point cloud (6x8), (12x16), (24x32), (48x64)
        scene_valid_mask = scene_valid_mask.view(N, L, 1, H, W).expand(N, L, 3, H, W)
        pt_8, pt_feat_8 = self.build_point_cloud_feat(scene_coord, scene_valid_mask, scene_feats[0])
        pt_16, pt_feat_16 = self.build_point_cloud_feat(scene_coord, scene_valid_mask, scene_feats[1])
        pt_32, pt_feat_32 = self.build_point_cloud_feat(scene_coord, scene_valid_mask, scene_feats[2])
        pt_64, pt_feat_64 = self.build_point_cloud_feat(scene_coord, scene_valid_mask, scene_feats[3])

        # compute feat_bot and corresponding coordinates
        # scene_rgb_feat_bot_C = scene_rgb_feats[3].size(2)
        # query_rgb_feat_bot_C = query_rgb_feats[3].size(1)
        _, _, scene_rgb_feat_bot_C, scene_bot_H, scene_bot_W = scene_rgb_feats[1].shape
        _, query_rgb_feat_bot_C, query_bot_H, query_bot_W = query_rgb_feats[1].shape

        _, _, corres_feat_H, corres_feat_W = query_rgb_feats[3].shape
        kernel_size = (corres_feat_H // query_bot_H, corres_feat_W // query_bot_W)
        # scene_rgb_feat_bot = F.unfold(
        #                 scene_rgb_feats[3].contiguous().view(N*L, scene_rgb_feat_bot_C, corres_feat_H, corres_feat_W),
        #                 kernel_size=kernel_size,
        #                 stride=kernel_size
        #                 ).view(N, L, -1, scene_bot_H, scene_bot_W)                     # dim:(N, L, 512*4*4, 3, 4)
        # query_rgb_feat_bot = F.unfold(
        #                 query_rgb_feats[3],
        #                 kernel_size=kernel_size,
        #                 stride=kernel_size
        #                 ).view(N, -1, query_bot_H, query_bot_W)                       # dim:(N, 512*4*4, 3, 4)
        scene_rgb_feat_bot = scene_rgb_feats[1].contiguous()                            # dim:(N, L, 512, 3, 4)
        query_rgb_feat_bot = query_rgb_feats[1].contiguous()                            # dim:(N, 512, 3, 4)
        scene_coord_feat_bot, scene_coord_mask_bot = self.build_point_cloud_feat(scene_coord, scene_valid_mask,
                                                                                 scene_rgb_feat_bot, shuffle=False)
        scene_coord_mask_bot = scene_coord_mask_bot[:, :, -1:, :, :]
        # dim:(N, L, 3, 3, 4)

        # [Core] All scene feature Match -------------------------------------------------------------------------------
        scene_rgb_feat_bot = scene_rgb_feat_bot.permute(0, 2, 1, 3, 4).contiguous().view(N, -1, 1, L*scene_bot_H*scene_bot_W)
        query_rgb_feat_bot = query_rgb_feat_bot.view(N, -1, query_bot_H*query_bot_W, 1)
        scene_coord_mask_bot = scene_coord_mask_bot.permute(0, 2, 1, 3, 4).contiguous().view(N, -1, 1, L * scene_bot_H * scene_bot_W)
        scene_coord_feat_bot = scene_coord_feat_bot.permute(0, 2, 1, 3, 4).\
            contiguous().view(N, 3, 1, L * scene_bot_H * scene_bot_W)                           # dim:(N, 3, 1, L*3*4)
        # corres_features = F.cosine_similarity(scene_rgb_feat_bot, query_rgb_feat_bot, dim=1, eps=1e-08).unsqueeze(1)
        # # sq_dot = scene_rgb_feat_bot * query_rgb_feat_bot                                    # dim:(N, 512, 3*4, L*3*4)
        # # corres_features = torch.sum(sq_dot, dim=1, keepdim=True)                            # dim:(N, 1, 3*4, L*3*4)
        # # weights = F.softmax(corres_features, dim=-1)                                        # dim:(N, 1, 3*4, L*3*4)
        # corres_features_topk, idx_topk = torch.topk(corres_features, 8, dim=-1)               # dim:(N, 1, 3*4, K)
        # weights_topk = F.normalize(corres_features_topk, p=1, dim=-1, eps=1e-12)              # dim:(N, 1, 3*4, K)
        # weights = torch.zeros_like(corres_features).scatter_(-1, idx_topk, weights_topk)      # dim:(N, 1, 3*4, L*3*4)
        out0, confid0 = self.attention(query_rgb_feat_bot, scene_rgb_feat_bot, scene_coord_feat_bot, mask=scene_coord_mask_bot)
        # dim:(N, 3, 3*4)

        if query_X_world is not None:
            diff = query_X_world[0].view(N, 3, query_bot_H * query_bot_W, 1) - scene_coord_feat_bot  # dim:(N, 3, 3*4, L*3*4)
            dist = torch.norm(diff, p=2, dim=1, keepdim=True)                                     # dim:(N, 1, 3*4, L*3*4)
            dist_topk, dist_idx_topk = torch.topk(dist, 8, dim=-1, largest=False)                 # dim:(N, 1, 3*4, K)
            weights0_gt = torch.zeros_like(dist).scatter_(-1, dist_idx_topk, torch.ones_like(dist_topk))  # dim:(N, 1, 3*4, L*3*4)
            weights0 = confid0                                                                    # dim:(N, 1, 3*4, L*3*4)
            weights0_gt = weights0_gt.squeeze(1)                                                  # dim:(N, 3*4, L*3*4)
            # weights0 = weights0.squeeze(1)                                                        # dim:(N, 3*4, L*3*4)
        else:
            weights0_gt = None
            weights0 = None

        # scene_coord_feat_bot = torch.sum(scene_coord_feat_bot * weights, dim=-1)                  # dim:(N, 3, 3*4)
        # out0 = scene_coord_feat_bot.view(N, 3, query_bot_H, query_bot_W)                          # dim:(N, 3, 3, 4)
        out0 = out0.view(N, 512, query_bot_H, query_bot_W)                                          # dim:(N, 512, 3, 4)

        # Conv to get rought prediction --------------------------------------------------------------------------------

        sq_feat_bot = torch.cat([query_rgb_feats[1], out0], dim=1)
        rough_res_feat = self.res_blocks(sq_feat_bot)
        out0 = self.rough_outconv(rough_res_feat)                                                    # dim:(N, 3, 3, 4)

        # compute refined query scene coordinates ----------------------------------------------------------------------
        rough_res_feat = F.interpolate(rough_res_feat, scale_factor=2, mode='nearest')
        out0_up = F.interpolate(out0, scale_factor=2, mode='nearest')#, align_corners=True)
        out0_up = out0_up.view(N, 3, -1).permute(0, 2, 1).contiguous()                          # (N, 6*8, 3)
        out1, rough_res_feat, weights1, weights1_gt = \
            self.refine_net0(pt_8, out0_up, pt_feat_8, query_rgb_feats[2], rough_res_feat,
                             query_X_world[1] if query_X_world is not None else None)           # (N, 3, 6, 8)

        rough_res_feat = F.interpolate(rough_res_feat, scale_factor=2, mode='nearest')
        out1_up = F.interpolate(out1, scale_factor=2, mode='nearest')#, align_corners=True)
        out1_up = out1_up.view(N, 3, -1).permute(0, 2, 1).contiguous()                          # (N, 12*16, 3)
#         rough_res_feat = F.interpolate(rough_res_feat, scale_factor=2, mode='bilinear', align_corners=True)
        out2, rough_res_feat, weights2, weights2_gt = \
            self.refine_net1(pt_16, out1_up, pt_feat_16, query_rgb_feats[3], rough_res_feat,
                             query_X_world[2] if query_X_world is not None else None)           # (N, 3, 12, 16)

        rough_res_feat = F.interpolate(rough_res_feat, scale_factor=2, mode='nearest')
        out2_up = F.interpolate(out2, scale_factor=2, mode='nearest')#, align_corners=True)
        out2_up = out2_up.view(N, 3, -1).permute(0, 2, 1).contiguous()                          # (N, 24x32, 3)
#         rough_res_feat = F.interpolate(rough_res_feat, scale_factor=2, mode='bilinear', align_corners=True)
        out3, rough_res_feat, weights3, weights3_gt = \
            self.refine_net2(pt_32, out2_up, pt_feat_32, query_rgb_feats[4], rough_res_feat,
                             query_X_world[3] if query_X_world is not None else None)           # (N, 3, 24, 32)

        rough_res_feat = F.interpolate(rough_res_feat, scale_factor=2, mode='nearest')
        out3_up = F.interpolate(out3, scale_factor=2, mode='nearest')#, align_corners=True)
        out3_up = out3_up.view(N, 3, -1).permute(0, 2, 1).contiguous()                          # (N, 48x64, 3)
#         rough_res_feat = F.interpolate(rough_res_feat, scale_factor=2, mode='bilinear', align_corners=True)
        out4, rough_res_feat, weights4, weights4_gt = \
            self.refine_net3(pt_64, out3_up, pt_feat_64, query_rgb_feats[5], rough_res_feat,
                             query_X_world[4] if query_X_world is not None else None)           # (N, 3, 48, 64)

        return [out0, out1, out2, out3, out4], [weights0, weights1, weights2, weights3, weights4],\
               [weights0_gt, weights1_gt, weights2_gt, weights3_gt, weights4_gt]

    def save_net_def(self, dir):
        extractor_path = inspect.getfile(RGBNet)
        shutil.copy(extractor_path, dir)
        refine_path = inspect.getfile(RefineLayer)
        shutil.copy(refine_path, dir)
        shutil.copy(os.path.realpath(__file__), dir)

    def add_to_scene_cache(self, scene, scene_valid_mask):
        """
        :param scene: dim: (1, L, 6, 192, 256), first 3 channel is rgb, second 3 channel is coords
        :param scene_valid_mask: dim: (192, 256)
        """
        N, L, _, H, W = scene.shape
        assert N == 1

        scene_rgb = scene[:, :, :3, :, :]
        scene_coord = scene[:, :, 3:, :, :]

        # Compute RGB features for scene -------------------------------------------------------------------------------
        scene_rgb_feats = self.rgb_net(scene_rgb.view(N * L, -1, H, W))  # dim: [(N*L, C, h, w)]
        scene_rgb_feats = [f.view(N, L, f.size(1), f.size(2), f.size(3))
                           for f in scene_rgb_feats]  # dim: [(N, L, C, h, w)]

        # Compute Coords features for scene ----------------------------------------------------------------------------
        # scene_coord_feats = self.coord_net(scene_coord.view(N * L, -1, H, W))  # dim: [(N*L, C, h, w)]
        # scene_coord_feats = [f.view(N, L, f.size(1), f.size(2), f.size(3))
        #                      for f in scene_coord_feats]  # dim: [(N, L, C, h, w)]

        # Build point clouds with feature ------------------------------------------------------------------------------
        # concatenate rgb & scene coord feature
        scene_feats = [scene_rgb_feats[i].contiguous() for i in [2, 3, 4, 5]]                            # dim:(N, L, c+c*, h, w)

        # compute scene local features point cloud (8x8), (16x16), (32x32)
        scene_valid_mask = scene_valid_mask.view(N, L, 1, H, W).expand(N, L, 3, H, W)
        pt_8, pt_feat_8 = self.build_point_cloud_feat(scene_coord, scene_valid_mask, scene_feats[0])
        pt_16, pt_feat_16 = self.build_point_cloud_feat(scene_coord, scene_valid_mask, scene_feats[1])
        pt_32, pt_feat_32 = self.build_point_cloud_feat(scene_coord, scene_valid_mask, scene_feats[2])
        pt_64, pt_feat_64 = self.build_point_cloud_feat(scene_coord, scene_valid_mask, scene_feats[3])
        scene_coord_feat_bot, scene_coord_mask_bot = self.build_point_cloud_feat(scene_coord, scene_valid_mask, scene_rgb_feats[1],
                                                              shuffle=False)
        # dim:(N, L, 3, 3, 4)
        scene_coord_mask_bot = scene_coord_mask_bot[:, :, -1:, :, :]

        # Add to scene cache -------------------------------------------------------------------------------------------
        self.scene_rgb_feats = [torch.cat([self.scene_rgb_feats[i], scene_rgb_feats[i]], dim=1)
                                for i in range(len(scene_rgb_feats))] \
            if self.scene_rgb_feats is not None else scene_rgb_feats
        self.scene_coord_feat_bot = torch.cat([self.scene_coord_feat_bot, scene_coord_feat_bot], dim=1) \
            if self.scene_coord_feat_bot is not None else scene_coord_feat_bot
        self.scene_coord_mask_bot = torch.cat([self.scene_coord_mask_bot, scene_coord_mask_bot], dim=1) \
            if self.scene_coord_mask_bot is not None else scene_coord_mask_bot
        # self.scene_coord_feats = [torch.cat([self.scene_coord_feats[i], scene_coord_feats[i]], dim=1)
        #                           for i in range(len(scene_coord_feats))] \
        #     if self.scene_coord_feats is not None else scene_coord_feats
        self.pt_8 = torch.cat([self.pt_8, pt_8[:, 1:, :]], dim=1) if self.pt_8 is not None else pt_8
        self.pt_feat_8 = torch.cat([self.pt_feat_8, pt_feat_8[:, :, 1:]],
                                   dim=2) if self.pt_feat_8 is not None else pt_feat_8

        self.pt_16 = torch.cat([self.pt_16, pt_16[:, 1:, :]], dim=1) if self.pt_16 is not None else pt_16
        self.pt_feat_16 = torch.cat([self.pt_feat_16, pt_feat_16[:, :, 1:]],
                                    dim=2) if self.pt_feat_16 is not None else pt_feat_16

        self.pt_32 = torch.cat([self.pt_32, pt_32[:, 1:, :]], dim=1) if self.pt_32 is not None else pt_32
        self.pt_feat_32 = torch.cat([self.pt_feat_32, pt_feat_32[:, :, 1:]],
                                    dim=2) if self.pt_feat_32 is not None else pt_feat_32

        self.pt_64 = torch.cat([self.pt_64, pt_64[:, 1:, :]], dim=1) if self.pt_64 is not None else pt_64
        self.pt_feat_64 = torch.cat([self.pt_feat_64, pt_feat_64[:, :, 1:]],
                                    dim=2) if self.pt_feat_64 is not None else pt_feat_64

    def clear_scene_cache(self):
        self.scene_rgb_feats = None
        self.scene_coord_feat_bot = None
        self.scene_coord_mask_bot = None
        # self.scene_coord_feats = None
        self.pt_8 = self.pt_feat_8 = None
        self.pt_16 = self.pt_feat_16 = None
        self.pt_32 = self.pt_feat_32 = None
        self.pt_64 = self.pt_feat_64 = None

    def shuffle_scene_point_cloud(self, in_pt_pos, in_pt_feat):
        # Remove the fake point ----------------------------------------------------------------------------------------
        pt_pos = in_pt_pos[:, 1:, :]  # (N, M, 3)
        pt_feat = in_pt_feat[:, :, 1:]  # (N, C, M)
        pt_pos = pt_pos.permute(1, 0, 2).contiguous()  # (M, N, 3)
        pt_feat = pt_feat.permute(2, 0, 1).contiguous()  # (M, N, C)
        M, N, C = pt_feat.shape

        # Random shuffle point cloud -----------------------------------------------------------------------------------
        pt_pos_feat = torch.cat([pt_pos, pt_feat], dim=-1)  # (M, N, 3+C)
        perm = torch.randperm(M)
        pt_pos_feat = pt_pos_feat[perm]  # (M, N, 3+C)
        pt_pos = pt_pos_feat[:, :, :3].permute(1, 0, 2).contiguous()  # (N, M, 3)
        pt_feat = pt_pos_feat[:, :, 3:].permute(1, 2, 0).contiguous()  # (N, C, M)

        # Add fake point -----------------------------------------------------------------------------------------------
        fake_point = torch.full((N, 1, 3), 999.0).to(pt_pos.device)
        pt_pos = torch.cat([fake_point, pt_pos], dim=1)
        fake_pt_feat = torch.zeros(N, pt_feat.size(1), 1).to(pt_feat.device)
        pt_feat = torch.cat([fake_pt_feat, pt_feat], dim=2)

        return pt_pos, pt_feat

    def prepare_query(self):
        assert self.pt_8 is not None, 'Please cache the scene first.'

        self.pt_8, self.pt_feat_8 = self.shuffle_scene_point_cloud(self.pt_8, self.pt_feat_8)
        self.pt_16, self.pt_feat_16 = self.shuffle_scene_point_cloud(self.pt_16, self.pt_feat_16)
        self.pt_32, self.pt_feat_32 = self.shuffle_scene_point_cloud(self.pt_32, self.pt_feat_32)
        self.pt_64, self.pt_feat_64 = self.shuffle_scene_point_cloud(self.pt_64, self.pt_feat_64)

    def query_forward(self, query_rgb, query_X_world=None):
        """
        :param query_rgb: dim: (3, 192, 256)
        """
        assert self.pt_8 is not None, 'Please cache the scene first.'

        # Get cached scene ---------------------------------------------------------------------------------------------
        scene_rgb_feats = self.scene_rgb_feats
        scene_coord_feat_bot = self.scene_coord_feat_bot
        scene_coord_mask_bot = self.scene_coord_mask_bot
        # scene_coord_feats = self.scene_coord_feats
        pt_8 = self.pt_8
        pt_feat_8 = self.pt_feat_8
        pt_16 = self.pt_16
        pt_feat_16 = self.pt_feat_16
        pt_32 = self.pt_32
        pt_feat_32 = self.pt_feat_32
        pt_64 = self.pt_64
        pt_feat_64 = self.pt_feat_64

        # Compute RGB feature for query --------------------------------------------------------------------------------
        query_rgb_feats = self.rgb_net(query_rgb)  # dim: [(N, C, h, w)]
        N = query_rgb.shape[0]

        # [Core] All scene feature Match -------------------------------------------------------------------------------
        # compute feat_bot and corresponding coordinates
        _, L, scene_rgb_feat_bot_C, scene_bot_H, scene_bot_W = scene_rgb_feats[1].shape
        _, query_rgb_feat_bot_C, query_bot_H, query_bot_W = query_rgb_feats[1].shape

        _, _, corres_feat_H, corres_feat_W = query_rgb_feats[3].shape
        scene_rgb_feat_bot = scene_rgb_feats[1].contiguous()                            # dim:(N, L, 512, 3, 4)
        query_rgb_feat_bot = query_rgb_feats[1].contiguous()                            # dim:(N, 512, 3, 4)

        # [Core] All scene feature Match -------------------------------------------------------------------------------
        scene_rgb_feat_bot = scene_rgb_feat_bot.permute(0, 2, 1, 3, 4).contiguous().view(N, -1, 1, L*scene_bot_H*scene_bot_W)
        query_rgb_feat_bot = query_rgb_feat_bot.view(N, -1, query_bot_H*query_bot_W, 1)
        scene_coord_mask_bot = scene_coord_mask_bot.permute(0, 2, 1, 3, 4).contiguous().view(N, -1, 1,
                                                                                             L * scene_bot_H * scene_bot_W)
        scene_coord_feat_bot = scene_coord_feat_bot.permute(0, 2, 1, 3, 4). \
            contiguous().view(N, 3, 1, L * scene_bot_H * scene_bot_W)  # dim:(N, 3, 1, L*3*4)
        # weights = self.attention(query_rgb_feat_bot, scene_rgb_feat_bot, scene_coord_mask_bot)                        # dim:(N, 1, 3*4, L*3*4)
        out0, confid0 = self.attention(query_rgb_feat_bot, scene_rgb_feat_bot, scene_coord_feat_bot,
                                       mask=scene_coord_mask_bot)

        # print(scene_coord_feat_bot.shape)
        if query_X_world is not None:
            diff = query_X_world[0].view(N, 3, query_bot_H * query_bot_W, 1) - scene_coord_feat_bot  # dim:(N, 3, 3*4, L*3*4)
            dist = torch.norm(diff, p=2, dim=1, keepdim=True)                                     # dim:(N, 1, 3*4, L*3*4)
            dist_topk, dist_idx_topk = torch.topk(dist, 8, dim=-1, largest=False)                 # dim:(N, 1, 3*4, K)
            weights0_gt = torch.zeros_like(dist).scatter_(-1, dist_idx_topk, torch.ones_like(dist_topk))  # dim:(N, 1, 3*4, L*3*4)
            weights0 = confid0                                                            # dim:(N, 1, 3*4, L*3*4)
            weights0_gt = weights0_gt.squeeze(1)                                                  # dim:(N, 3*4, L*3*4)
            # weights0 = weights0.squeeze(1)                                                        # dim:(N, 3*4, L*3*4)
        else:
            weights0_gt = None
            weights0 = None

        # scene_coord_feat_bot = torch.sum(scene_coord_feat_bot * weights, dim=-1)                  # dim:(N, 3, 3*4)
        # out0 = scene_coord_feat_bot.view(N, 3, query_bot_H, query_bot_W)                          # dim:(N, 3, 3, 4)
        out0 = out0.view(N, 512, query_bot_H, query_bot_W)  # dim:(N, 3, 3, 4)

        # Conv to get rought prediction --------------------------------------------------------------------------------

        sq_feat_bot = torch.cat([query_rgb_feats[1], out0], dim=1)
        rough_res_feat = self.res_blocks(sq_feat_bot)
        out0 = self.rough_outconv(rough_res_feat)                                                    # dim:(N, 3, 3, 4

        # compute refined query scene coordinates ----------------------------------------------------------------------
        rough_res_feat = F.interpolate(rough_res_feat, scale_factor=2, mode='nearest')
        out0_up = F.interpolate(out0, scale_factor=2, mode='nearest')  # , align_corners=True)
        out0_up = out0_up.view(N, 3, -1).permute(0, 2, 1).contiguous()  # (N, 6*8, 3)
        out1, rough_res_feat, weights1, weights1_gt = \
            self.refine_net0(pt_8, out0_up, pt_feat_8, query_rgb_feats[2], rough_res_feat,
                             query_X_world[1] if query_X_world is not None else None)  # (N, 3, 6, 8)

        rough_res_feat = F.interpolate(rough_res_feat, scale_factor=2, mode='nearest')
        out1_up = F.interpolate(out1, scale_factor=2, mode='nearest')  # , align_corners=True)
        out1_up = out1_up.view(N, 3, -1).permute(0, 2, 1).contiguous()  # (N, 12*16, 3)
        #         rough_res_feat = F.interpolate(rough_res_feat, scale_factor=2, mode='bilinear', align_corners=True)
        out2, rough_res_feat, weights2, weights2_gt = \
            self.refine_net1(pt_16, out1_up, pt_feat_16, query_rgb_feats[3], rough_res_feat,
                             query_X_world[2] if query_X_world is not None else None)  # (N, 3, 12, 16)

        rough_res_feat = F.interpolate(rough_res_feat, scale_factor=2, mode='nearest')
        out2_up = F.interpolate(out2, scale_factor=2, mode='nearest')  # , align_corners=True)
        out2_up = out2_up.view(N, 3, -1).permute(0, 2, 1).contiguous()  # (N, 24x32, 3)
        #         rough_res_feat = F.interpolate(rough_res_feat, scale_factor=2, mode='bilinear', align_corners=True)
        out3, rough_res_feat, weights3, weights3_gt = \
            self.refine_net2(pt_32, out2_up, pt_feat_32, query_rgb_feats[4], rough_res_feat,
                             query_X_world[3] if query_X_world is not None else None)  # (N, 3, 24, 32)

        rough_res_feat = F.interpolate(rough_res_feat, scale_factor=2, mode='nearest')
        out3_up = F.interpolate(out3, scale_factor=2, mode='nearest')  # , align_corners=True)
        out3_up = out3_up.view(N, 3, -1).permute(0, 2, 1).contiguous()  # (N, 48x64, 3)
        #         rough_res_feat = F.interpolate(rough_res_feat, scale_factor=2, mode='bilinear', align_corners=True)
        out4, rough_res_feat, weights4, weights4_gt = \
            self.refine_net3(pt_64, out3_up, pt_feat_64, query_rgb_feats[5], rough_res_feat,
                             query_X_world[4] if query_X_world is not None else None)  # (N, 3, 48, 64)

        return [out0, out1, out2, out3, out4], [weights0, weights1, weights2, weights3, weights4],\
               [weights0_gt, weights1_gt, weights2_gt, weights3_gt, weights4_gt]

    def bot1x1_forward(self, frames_rgb):
        N, C, H, W = frames_rgb.shape
        if C != 3:
            scene_rgb = frames_rgb[:, :3, :, :]
        else:
            scene_rgb = frames_rgb

        # Compute RGB features -----------------------------------------------------------------------------------------
        scene_rgb_feats = self.rgb_net(scene_rgb)                                           # dim: [(N, C, h, w)]
        _, scene_rgb_feat_bot_C, scene_bot_H, scene_bot_W = scene_rgb_feats[1].shape
        scene_rgb_feat_bot = scene_rgb_feats[1]                                             # dim:(N, C, H, W)
        # scene_rgb_feat_bot = scene_rgb_feat_bot.permute(0, 2, 1, 3, 4). \
        #     contiguous().view(1, scene_rgb_feat_bot_C, 1, scene_bot_H * scene_bot_W)

        return scene_rgb_feat_bot.view(N, -1)


if __name__ == '__main__':
    with torch.cuda.device(0):
        model = Corres2D3DNet()
        model.cuda()
        rand_scene_input = torch.rand(1, 5, 6, 192, 256).cuda()
        rand_query_input = torch.rand(1, 3, 192, 256).cuda()
        rand_scene_valid_mask = torch.ByteTensor(1, 5, 192, 256).random_(0, 2).cuda()
        rand_query_X_world = [torch.rand(1, 3, 3*(2**i), 4*(2**i)).cuda() for i in range(5)]
        x = model.forward(rand_query_input, rand_scene_input, rand_scene_valid_mask, rand_query_X_world)
        print(x[0][0].shape, x[0][1].shape, x[0][2].shape, x[0][3].shape, x[0][4].shape)
        print(x[0][0].max())
        # print(x[1][0].shape, x[1][1].shape, x[1][2].shape, x[1][3].shape, x[1][4].shape)
        print(x[2][0].shape, x[2][1].shape, x[2][2].shape, x[2][3].shape, x[2][4].shape)
        # x = x.view(x.size(0), -1)

        # model = Corres3DEncoderDecoder()
        # model.cuda()
        # model.summary()
        # x = model.forward(torch.rand(1, 6, 256, 256).cuda())
        # print(x[0][0].shape, x[0][1].shape, x[0][2].shape, x[0][3].shape)
