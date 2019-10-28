import torch
import torch.nn as nn
import torch.nn.functional as F
from relocal.vlad_module import NetVLAD
from torchvision.models import resnet18
import relocal.backbone_drn as drn
from core_dl.base_net import BaseNet
import inspect
import shutil
import os
from relocal.pointnet2.pointnet2_utils.pointnet2_modules import *


if __name__ == "__main__":
    from torch.autograd import Variable
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = Variable(torch.randn(2, 9, 3).cuda(), requires_grad=True)
    xyz_feats = Variable(torch.randn(2, 9, 6).cuda(), requires_grad=True)

    test_module = PointnetSAModuleMSG(
        npoint=2, radii=[5.0, 10.0], nsamples=[6, 3], mlps=[[9, 3], [9, 6]]
    )
    test_module.cuda()
    print(test_module(xyz, xyz_feats))

    #  test_module = PointnetFPModule(mlp=[6, 6])
    #  test_module.cuda()
    #  from torch.autograd import gradcheck
    #  inputs = (xyz, xyz, None, xyz_feats)
    #  test = gradcheck(test_module, inputs, eps=1e-6, atol=1e-4)
    #  print(test)

    for _ in range(1):
        _, new_features = test_module(xyz, xyz_feats)
        new_features.backward(
            torch.cuda.FloatTensor(*new_features.size()).fill_(1)
        )
        print(new_features)
        print(xyz.grad)

