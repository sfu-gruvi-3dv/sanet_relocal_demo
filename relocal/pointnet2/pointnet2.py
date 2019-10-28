import torch
import torch.nn as nn
import numpy as np
from .pointnet2_utils import pytorch_utils as pt_utils
from .pointnet2_utils.pointnet2_modules import (
    PointnetSAModuleMSG, PointnetSAModule
)
from collections import namedtuple


class Pointnet2test(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Classification network

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, input_channels=3, use_xyz=False):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[input_channels, 128, 128,
                       256], [input_channels, 256, 256, 512], ],
                use_xyz=use_xyz
            )
        )

        input_channels = 256 + 512
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=32,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[input_channels, 512, 512,
                       1024], [input_channels, 512, 512, 1024], ],
                use_xyz=use_xyz
            )
        )

        input_channels = 1024 + 1024
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=8,
                radii=[0.4, 0.8],
                nsamples=[16, 32],
                mlps=[[input_channels, 1024, 1024,
                       2048], [input_channels, 1024, 1024, 2048], ],
                use_xyz=use_xyz
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                mlp=[2048 + 2048, 2048, 2048, 2048], use_xyz=use_xyz
            )
        )

        self.FC_layer = nn.Sequential(
            pt_utils.FC(2048, 512, bn=True),
            nn.Dropout(p=0.5),
            pt_utils.FC(512, 256, bn=True),
            nn.Dropout(p=0.5),
            pt_utils.FC(256, num_classes, activation=None)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:6].contiguous()
        ## OK, I will add random 3 dim as camera pose as simulation
        # res = torch.from_numpy(np.random.rand(*xyz.shape)).type_as(xyz)
        # res = torch.from_numpy(np.zeros(xyz.shape)).type_as(xyz)
        # xyz = torch.cat([xyz, res], dim=-1).contiguous()
        # res = torch.from_numpy(np.zeros(xyz.shape[:-1]+(128,))).type_as(xyz)
        # pc = torch.cat([pc, res], dim=-1)
        features = (
            pc[..., 6:].transpose(1, 2).contiguous()
            if pc.size(-1) > 6 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.FC_layer(features.squeeze(-1))


class scene_query(nn.Module):

    def __init__(self, num_classes=6, scene_dim=256, fea_dim=512, input_channels=512, use_xyz=True):
        super().__init__()
        self.pointnet2 = Pointnet2test(num_classes=scene_dim, input_channels=input_channels, use_xyz=use_xyz)
        self.FC_layer = nn.Sequential(
            pt_utils.FC(scene_dim + fea_dim, 512, bn=True),
            nn.Dropout(p=0.5),
            pt_utils.FC(512, 256, bn=True),
            nn.Dropout(p=0.5),
            pt_utils.FC(256, num_classes, activation=None)
        )
        pass

    def forward(self, pose, fea, img):
        in_feat = torch.cat([pose, fea], dim=-1)  # shape=(2, 40, 6+512)
        scene = self.Pointnet2(in_feat)  # shape=(2, 256)
        img = torch.cat([img, scene], -1)
        output = self.FC_layer(img)
        return output
