import os, sys, warnings, pickle, argparse, shutil, inspect, random, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from banet_track.ba_module import x_2d_coords_torch, batched_pi_inv, batched_transpose, batched_inv_pose, batched_pi
import core_3dv.camera_operator as cam_opt
from core_math.transfom import quaternion_about_axis, quaternion_matrix


def rescale_scene_coords(query_coords:list,
                         scene_coords:torch.Tensor,
                         neg_tags:torch.Tensor,
                         rescale_dist=4.0, use_query=False):

    _, _, h, w = query_coords[4].shape
    N, L, C, H, W = scene_coords.shape

    if use_query:
        max_axis = torch.max(query_coords[4].view(N, 3, h * w), dim=2)[0]  # dim (N, 3)
        min_axis = torch.min(query_coords[4].view(N, 3, h * w), dim=2)[0]  # dim (N, 3)
        rect = max_axis - min_axis  # dim (N, 3)
        max_dim = torch.max(rect, dim=1)[0]  # dim (N, 1)
        rescale_factor = rescale_dist / (max_dim + 1e-5)
    else:
        scene_coords = scene_coords.view((N, L, 3, H*W))

        max_axis = torch.max(scene_coords, dim=3)[0]                                     # dim (N, L, 3)
        min_axis = torch.min(scene_coords, dim=3)[0]                                     # dim (N, L, 3)
        rect = max_axis - min_axis                                                       # dim (N, L, 3)
        rect = rect.permute(0, 2, 1)                                                     # dim (N, 3, L)
        max_rect = torch.max(rect, dim=2)[0]                                             # dim (N, 3)
        max_dim = torch.max(max_rect, dim=1)[0]                                          # dim (N, 1)
        rescale_factor = rescale_dist / (max_dim + 1e-5)

        # rescale query and scene
        scene_coords = scene_coords.view((N, L, C, H, W))

    scene_coords = rescale_factor.view((N, 1, 1, 1, 1)) * scene_coords
    for q_idx in range(len(query_coords)):
        query_coords[q_idx] = rescale_factor.view((N, 1, 1, 1)) * query_coords[q_idx]

    return query_coords, scene_coords, rescale_factor.view(N)

def preprocess(sample_dict, pre_x2d, out_dim, rescale_dist=0.0):
    rand_angle = np.random.random_sample() * 2.0 * np.pi
    rand_R = quaternion_matrix(quaternion_about_axis(rand_angle, (0.0, 1.0, 0.0)))[:3, :3]
    rand_R = torch.FloatTensor(rand_R).unsqueeze(0)

    scene_rgb = sample_dict['frames_img'][:, :5, ...].cuda()
    scene_depth = sample_dict['frames_depth'][:, :5, ...].cuda()
    scene_K = sample_dict['frames_K'][:, :5, ...].cuda()
    scene_Tcw = sample_dict['frames_Tcw'][:, :5, ...]
    scene_ori_rgb = sample_dict['frames_ori_img'][:, :5, ...].cuda()
    scene_neg_tags = sample_dict['frames_neg_tags'][:, :5, ...].cuda()

    N, L, C, H, W = scene_rgb.shape
    # scene_rgb = scene_rgb.view(N, L, C, H, W)
    scene_depth = scene_depth.view(N * L, 1, H, W)
    scene_K = scene_K.view(N * L, 3, 3)
    scene_Tcw = scene_Tcw.view(N * L, 3, 4)

    # generate 3D world position of scene
    d = scene_depth.view(N * L, H * W, 1)  # dim (N*L, H*W, 1)
    X_3d = batched_pi_inv(scene_K, pre_x2d, d)  # dim (N*L, H*W, 3)
    Rwc, twc = batched_inv_pose(R=scene_Tcw[:, :3, :3],
                                t=scene_Tcw[:, :3, 3].squeeze(-1))  # dim (N*L, 3, 3), （N, 3)
    X_world = batched_transpose(Rwc.cuda(), twc.cuda(), X_3d)  # dim (N*L, H*W, 3)
    X_world = X_world.contiguous().view(N, L * H * W, 3)        # dim (N, L*H*W, 3)
    scene_center = torch.mean(X_world, dim=1)  # dim (N, 3)
    X_world -= scene_center.view(N, 1, 3)
    X_world = batched_transpose(rand_R.cuda().expand(N, 3, 3),
                                torch.zeros(1, 3, 1).cuda().expand(N, 3, 1),
                                X_world)  # dim (N, L*H*W, 3), data augmentation
    X_world = X_world.view(N, L, H, W, 3).permute(0, 1, 4, 2, 3).contiguous()  # dim (N, L, 3, H, W)

    # query image:
    query_img = sample_dict['img']
    query_ori_img = sample_dict['ori_img']

    # compute multiscale ground truth query_X_worlds & valid_masks
    query_X_worlds = []
    valid_masks = []
    out_H, out_W = out_dim
    query_depth = sample_dict['depth'].cuda()
    ori_query_depth = query_depth.clone()
    N, C, H, W = query_depth.shape
    for i in range(4):
        query_depth_patch = F.unfold(
            query_depth,
            kernel_size=(H // out_H, W // out_W),
            stride=(H // out_H, W // out_W)
        ).view(N, -1, out_H, out_W)
        mask = torch.gt(query_depth_patch, 1e-5)
        count = torch.sum(mask.float(), dim=1)
        query_depth_down = torch.sum(query_depth_patch * mask.float(), dim=1) / \
                           torch.where(torch.le(count, 1e-5),
                                       torch.full(count.shape, 1e6).to(count.device),
                                       count)  # (N, 1, out_H, out_W)
        query_Tcw = sample_dict['Tcw']
        query_K = sample_dict['K'].clone().cuda()
        query_K[:, 0, 0] *= out_W / W
        query_K[:, 0, 2] *= out_W / W
        query_K[:, 1, 1] *= out_H / H
        query_K[:, 1, 2] *= out_H / H
        query_d = query_depth_down.view(N, out_H * out_W, 1)  # dim (N, H*W, 1)
        out_x_2d = x_2d_coords_torch(N, out_H, out_W).cuda().view(N, -1, 2)
        query_X_3d = batched_pi_inv(query_K, out_x_2d, query_d)  # dim (N, H*W, 3)
        query_Rwc, query_twc = batched_inv_pose(R=query_Tcw[:, :3, :3],
                                                t=query_Tcw[:, :3, 3].squeeze(-1))  # dim (N, 3, 3), （N, 3)
        query_X_world = batched_transpose(query_Rwc.cuda(), query_twc.cuda(), query_X_3d)  # dim (N, H*W, 3)
        query_X_world -= scene_center.view(N, 1, 3)
        query_X_world = batched_transpose(rand_R.cuda().expand(N, 3, 3),
                                          torch.zeros(1, 3, 1).cuda().expand(N, 3, 1),
                                          query_X_world)  # dim (N, H*W, 3), data augmentation
        query_X_world = query_X_world.permute(0, 2, 1).view(N, 3, out_H, out_W).contiguous()  # dim (N, 3, H, W)
        query_X_worlds.append(query_X_world.cuda())

        valid_masks.append(torch.gt(query_depth_down, 1e-5).cuda().view(N, out_H, out_W))

        if i == 3:
            query_X_worlds.append(query_X_world.cuda())
            valid_masks.append(torch.gt(query_depth_down, 1e-5).cuda().view(N, out_H, out_W))

        out_H //= 2
        out_W //= 2

    # compute norm_query_Tcw for normalized scene coordinate
    query_twc = query_twc.cuda() - scene_center.view(N, 3, 1)
    norm_query_Twc = torch.cat([query_Rwc.cuda(), query_twc], dim=-1)  # dim (N, 3, 4)
    norm_query_Twc = torch.bmm(rand_R.cuda().expand(N, 3, 3), norm_query_Twc)  # dim (N, 3, 4)
    query_Rcw, query_tcw = batched_inv_pose(R=norm_query_Twc[:, :3, :3],
                                            t=norm_query_Twc[:, :3, 3].squeeze(-1))  # dim (N, 3, 3), （N, 3)
    norm_query_Tcw = torch.cat([query_Rcw, query_tcw.view(N, 3, 1)], dim=-1)  # dim (N, 3, 4)

    # compute down sampled query K
    out_H, out_W = out_dim
    query_K = sample_dict['K'].clone().cuda()
    query_K[:, 0, 0] *= out_W / W
    query_K[:, 0, 2] *= out_W / W
    query_K[:, 1, 1] *= out_H / H
    query_K[:, 1, 2] *= out_H / H

    if rescale_dist > 0:
        query_X_worlds, X_world, rescale_factor = rescale_scene_coords(query_X_worlds, X_world, scene_neg_tags, rescale_dist)
    else:
        rescale_factor = torch.ones(N)
    scene_input = torch.cat((scene_rgb, X_world), dim=2)

    return scene_input.cuda(), query_img.cuda(), query_X_worlds[::-1], valid_masks[::-1], \
           scene_ori_rgb.cuda(), query_ori_img.cuda(), X_world.cuda(), \
           torch.gt(scene_depth, 1e-5).cuda().view(N, L, H, W), norm_query_Tcw, query_K, scene_neg_tags, rescale_factor.cuda()


if __name__ == '__main__':
    from relocal_data.cambridge.cambridge_dataset import CambridgeDataset
    from torch.utils.data import DataLoader, Dataset

    with open('/home/ziqianb/Downloads/cambridge/KingsCollege/valid.bin', 'rb') as f:
        data_list = pickle.load(f)

    data_set = CambridgeDataset(seq_data_list=data_list,
                                cambridge_base_dir='/home/ziqianb/Downloads/cambridge',
                                transform=None, rand_flip=False, remove_depth_outlier_ratio=0.01,
                                output_dim=(3, 192, 256))

    # sample = data_set.__getitem__(12)
    data_loader = DataLoader(data_set, batch_size=2, num_workers=0, shuffle=False)
    print('size of the dataset:', len(data_set))

    x_2d = x_2d_coords_torch(2*5, 192, 256).view(2*5, -1, 2).cuda()
    seq_dict = next(iter(data_loader))
    scene_input, query_img, query_X_world, valid_mask, scene_ori_rgb, query_ori_img, \
    X_world, scene_valid_mask, norm_query_Tcw, query_K, s_neg_tags = preprocess(seq_dict, x_2d,
                                                                                    (192, 256), rescale_dist=4.0)
