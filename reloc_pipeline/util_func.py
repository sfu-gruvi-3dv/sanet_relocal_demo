import cv2, os, sys, pickle, argparse, shutil, inspect, random, time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision.utils import make_grid

import core_3dv.camera_operator as cam_opt
import core_dl.module_util as dl_util
from relocal_data.seven_scene.seven_scene_dict_preprocess import preprocess as manual_preprocess

from visualizer.visualizer_2d import show_multiple_img
from core_dl.train_params import TrainParameters
from core_dl.base_train_box import BaseTrainBox
from relocal.corres_net import Corres2D3DNet
from banet_track.ba_module import x_2d_coords_torch, batched_pi_inv, batched_transpose, batched_inv_pose, batched_pi
from core_math.transfom import random_rotation_matrix, random_vector, quaternion_about_axis, quaternion_matrix
from core_dl.torch_vision_ext import heatmap_blend, colormap
from evaluator.basic_metric import rel_rot_angle, rel_distance

from relocal.vlad_encoder import VLADEncoder
from relocal.point_feat import *

sys.path.append('./libs/lm_pnp/build')
import lm_pnp

def preprocess(sample_dict, x_2d, out_dim=(32, 32)):
    rand_angle = np.random.random_sample() * 2.0 * np.pi
    rand_R = quaternion_matrix(quaternion_about_axis(rand_angle, (0.0, 1.0, 0.0)))[:3, :3]
    rand_R = torch.FloatTensor(rand_R).unsqueeze(0)

    scene_rgb = sample_dict['frames_img'].cuda()
    scene_depth = sample_dict['frames_depth'].cuda()
    scene_K = sample_dict['frames_K'].cuda()
    scene_Tcw = sample_dict['frames_Tcw']
    scene_ori_rgb = sample_dict['frames_ori_img'].cuda()
    N, L, C, H, W = scene_rgb.shape
    # scene_rgb = scene_rgb.view(N, L, C, H, W)
    scene_depth = scene_depth.view(N * L, 1, H, W)
    scene_K = scene_K.view(N * L, 3, 3)
    scene_Tcw = scene_Tcw.view(N * L, 3, 4)

    # generate 3D world position of scene
    d = scene_depth.view(N * L, H * W, 1)  # dim (N*L, H*W, 1)
    X_3d = batched_pi_inv(scene_K, x_2d, d)  # dim (N*L, H*W, 3)
    Rwc, twc = batched_inv_pose(R=scene_Tcw[:, :3, :3],
                                t=scene_Tcw[:, :3, 3].squeeze(-1))  # dim (N*L, 3, 3), （N, 3)
    X_world = batched_transpose(Rwc.cuda(), twc.cuda(), X_3d)  # dim (N*L, H*W, 3)
    X_world = X_world.view(N, L * H * W, 3)  # dim (N, L*H*W, 3)
    scene_center = torch.mean(X_world, dim=1)  # dim (N, 3)
    X_world -= scene_center.view(N, 1, 3)
    X_world = batched_transpose(rand_R.cuda().expand(N, 3, 3),
                                torch.zeros(1, 3, 1).cuda().expand(N, 3, 1),
                                X_world)  # dim (N, L*H*W, 3), data augmentation
    X_world = X_world.view(N, L, H, W, 3).permute(0, 1, 4, 2, 3).contiguous()  # dim (N, L, 3, H, W)
    scene_input = torch.cat((scene_rgb, X_world), dim=2)

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
    for i in range(5):
        query_depth_patch = F.unfold(
            query_depth,
            kernel_size=(H // out_H, W // out_W),
            stride=(H // out_H, W // out_W)
        ).view(N, -1, out_H, out_W)
        mask = torch.gt(query_depth_patch, 1e-5)
        count = torch.sum(mask.float(), dim=1)
        query_depth_down = torch.sum(query_depth_patch * mask.float(), dim=1) /\
                           torch.where(torch.le(count, 1e-5),
                                       torch.full(count.shape, 1e6).cuda(),
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

#         if i == 3:
#             query_X_worlds.append(query_X_world.cuda())
#             valid_masks.append(torch.gt(query_depth_down, 1e-5).cuda().view(N, out_H, out_W))

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

    return scene_input.cuda(), query_img.cuda(), query_X_worlds[::-1], valid_masks[::-1], \
           scene_ori_rgb.cuda(), query_ori_img.cuda(), X_world.cuda(), \
           torch.gt(scene_depth, 1e-5).cuda().view(N, L, H, W), \
           scene_center.cuda(), query_Tcw.cuda(), query_K.cuda(), rand_R.expand(N, 3, 3).cuda()


def preprocess_scene(x_2d, scene_rgb, scene_depth, scene_K, scene_Tcw, scene_ori_rgb):
    rand_R = torch.eye(3).unsqueeze(0)
    N, L, C, H, W = scene_rgb.shape
    # scene_rgb = scene_rgb.view(N, L, C, H, W)
    scene_depth = scene_depth.view(N * L, 1, H, W)
    scene_K = scene_K.view(N * L, 3, 3)
    scene_Tcw = scene_Tcw.view(N * L, 3, 4)

    # generate 3D world position of scene
    d = scene_depth.view(N * L, H * W, 1)  # dim (N*L, H*W, 1)
    X_3d = batched_pi_inv(scene_K, x_2d, d)  # dim (N*L, H*W, 3)
    Rwc, twc = batched_inv_pose(R=scene_Tcw[:, :3, :3],
                                t=scene_Tcw[:, :3, 3].squeeze(-1))  # dim (N*L, 3, 3), （N, 3)
    X_world = batched_transpose(Rwc.cuda(), twc.cuda(), X_3d)  # dim (N*L, H*W, 3)
    X_world = X_world.view(N, L * H * W, 3)  # dim (N, L*H*W, 3)
    scene_center = torch.mean(X_world, dim=1)  # dim (N, 3)
    X_world -= scene_center.view(N, 1, 3)
    X_world = batched_transpose(rand_R.cuda().expand(N, 3, 3),
                                torch.zeros(1, 3, 1).cuda().expand(N, 3, 1),
                                X_world)  # dim (N, L*H*W, 3), data augmentation
    X_world = X_world.view(N, L, H, W, 3).permute(0, 1, 4, 2, 3).contiguous()  # dim (N, L, 3, H, W)
    scene_input = torch.cat((scene_rgb, X_world), dim=2)

    return scene_input.cuda(), scene_ori_rgb.cuda(), X_world.cuda(), \
           torch.gt(scene_depth, 1e-5).cuda().view(N, L, H, W), scene_center.cuda(), rand_R.expand(N, 3, 3).cuda()


def preprocess_query(query_img, query_depth, query_ori_img, query_Tcw, ori_query_K, scene_center, rand_R, out_dim=(48, 64)):
    # compute multiscale ground truth query_X_worlds & valid_masks
    query_X_worlds = []
    valid_masks = []
    out_H, out_W = out_dim
    ori_query_depth = query_depth.clone()
    N, C, H, W = query_depth.shape
    for i in range(5):
        query_depth_patch = F.unfold(
            query_depth,
            kernel_size=(H // out_H, W // out_W),
            stride=(H // out_H, W // out_W)
        ).view(N, -1, out_H, out_W)
        mask = torch.gt(query_depth_patch, 1e-5)
        count = torch.sum(mask.float(), dim=1)
        query_depth_down = torch.sum(query_depth_patch * mask.float(), dim=1) /\
                           torch.where(torch.le(count, 1e-5),
                                       torch.full(count.shape, 1e6).cuda(),
                                       count)  # (N, 1, out_H, out_W)
        query_K = ori_query_K.clone().cuda()
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

#         if i == 3:
#             query_X_worlds.append(query_X_world.cuda())
#             valid_masks.append(torch.gt(query_depth_down, 1e-5).cuda().view(N, out_H, out_W))

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
    query_K = ori_query_K.clone().cuda()
    query_K[:, 0, 0] *= out_W / W
    query_K[:, 0, 2] *= out_W / W
    query_K[:, 1, 1] *= out_H / H
    query_K[:, 1, 2] *= out_H / H

    return query_img.cuda(), query_X_worlds[::-1], valid_masks[::-1], query_ori_img.cuda(), \
           scene_center.cuda(), query_Tcw.cuda(), query_K.cuda(), rand_R.expand(N, 3, 3).cuda()


# Measurements
def flow_vis(flow):
    # Use Hue, Saturation, Value colour model
    hsv = np.zeros(flow.shape, dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return bgr


def recover_original_scene_coordinates(query_X_w, rand_R, scene_center):
    N, _, H, W = query_X_w.shape

    # recover original scene coordinates
    query_X_3d_w = query_X_w.permute(0, 2, 3, 1).view(N, -1, 3)
    rand_R_t = torch.transpose(rand_R, 1, 2).to(query_X_3d_w.device)
    query_X_3d_w = batched_transpose(rand_R_t, torch.zeros(N, 3).to(query_X_3d_w.device), query_X_3d_w)
    query_X_3d_w += scene_center.view(N, 1, 3)
    query_X_3d_w = query_X_3d_w.view(N, H, W, 3)

    return query_X_3d_w


def compute_pose_pnp_from_valid_pixels(gt_Tcws, query_X_w, rand_R, scene_center, query_K, valid_pix_idx, pnp_x_2d,
                                       repro_thres):
    N, _, H, W = query_X_w.shape

    # recover original scene coordinates
    query_X_3d_w = query_X_w.permute(0, 2, 3, 1).view(N, -1, 3)
    rand_R_t = torch.transpose(rand_R, 1, 2).to(query_X_3d_w.device)
    query_X_3d_w = batched_transpose(rand_R_t, torch.zeros(N, 3).to(query_X_3d_w.device), query_X_3d_w)
    query_X_3d_w += scene_center.view(N, 1, 3)
    query_X_3d_w = recover_original_scene_coordinates(query_X_w, rand_R, scene_center)
    query_X_3d_w = query_X_3d_w.view(N, H, W, 3).squeeze(0).detach().cpu().numpy()

    # select valid pixels with input index
    x, y = valid_pix_idx
    x_2d_valid = pnp_x_2d[y, x, :]
    query_X_3d_valid = query_X_3d_w[y, x, :]
    selected_pixels = query_X_3d_valid.shape[0]

    query_X_3d_valid = query_X_3d_valid.reshape(1, selected_pixels, 3)
    x_2d_valid = x_2d_valid.reshape(1, selected_pixels, 2)

    # run Ransac PnP
    dist = np.zeros(4)
    k = query_K.squeeze(0).detach().cpu().numpy()
    retval, R_res, t_res, ransc_inlier = cv2.solvePnPRansac(query_X_3d_valid, x_2d_valid, k, dist,
                                                            reprojectionError=repro_thres, )
    #     print(retval)
    #     _, R_res, t_res = cv2.solvePnP(query_X_3d_valid, x_2d_valid, k, dist)#, flags=cv2.SOLVEPNP_EPNP)

    R_res, _ = cv2.Rodrigues(R_res)
    pnp_pose = np.eye(4, dtype=np.float32)
    pnp_pose[:3, :3] = R_res
    pnp_pose[:3, 3] = t_res.ravel()

    # measure accuracy
    gt_pose = gt_Tcws.squeeze(0).detach().cpu().numpy()

    R_acc = rel_rot_angle(pnp_pose, gt_pose)
    t_acc = rel_distance(pnp_pose, gt_pose)

    #     ransc_inlier = None
    return R_acc, t_acc, pnp_pose, ransc_inlier


def compute_accuracy(pred, gt, valid_mask, threshold):
    """
    :param pred: (N, C, H, W)
    :param gt: (N, C, H, W)
    :param valid_mask: (N, H, W)
    :param threshold:
    :return:
    """
    pred = pred.detach()
    count = torch.sum(
        torch.masked_select(torch.lt(torch.norm(pred - gt, p=2, dim=1), threshold), valid_mask).long()
    )
    num_valid = torch.sum(valid_mask.long())
    if num_valid == 0:
        return torch.tensor(0.0)
    else:
        return count.float() / num_valid.float()


def accuracy_heatmap(ori_img, pred, gt, dist_range=(0, 1.0), frame_dim=(256, 256), out_dim=(32, 32)):
    """
    Generate accuracy heatmap
    :param ori_img: original image with dim: (N, 3, H, W)
    :param pred: the predicted scene coordinates, dim: (N, 3, h, w)
    :param gt: the ground-truth scene coordinates, dim: (N, 3, h, w)
    :param dist_range: the distance threshold range, (min, max)
    :return: the blended heatmap with dim: (N, 3, H, W)
    """
    pred = pred.detach()
    N, C1, H, W = pred.shape
    dist = torch.norm(pred - gt, p=2, dim=1).unsqueeze(1)
    dist = F.interpolate(dist,
                         scale_factor=(frame_dim[0] // out_dim[0],
                                       frame_dim[1] // out_dim[1]),
                         mode='nearest')
    blended_heatmap = heatmap_blend(ori_img, dist, heatmap_clip_range=dist_range)
    return blended_heatmap


def compute_pose_lm_pnp(gt_Tcws, query_X_w, rand_R, scene_center, query_K, pnp_x_2d, repro_thres):
    N, _, H, W = query_X_w.shape

    # recover original scene coordinates
    query_X_3d_w = query_X_w.permute(0, 2, 3, 1).view(N, -1, 3)
    rand_R_t = torch.transpose(rand_R, 1, 2).to(query_X_3d_w.device)
    query_X_3d_w = batched_transpose(rand_R_t, torch.zeros(N, 3).to(query_X_3d_w.device), query_X_3d_w)
    query_X_3d_w += scene_center.view(N, 1, 3)
    query_X_3d_w = recover_original_scene_coordinates(query_X_w, rand_R, scene_center)
    query_X_3d_w = query_X_3d_w.view(N, H, W, 3).squeeze(0).detach().cpu().numpy()

    # run Ransac PnP
    lm_pnp_pose_vec, inlier_map = lm_pnp.compute_lm_pnp(pnp_x_2d, query_X_3d_w, query_K, repro_thres, 128, 100)
    R_res, _ = cv2.Rodrigues(lm_pnp_pose_vec[:3])
    lm_pnp_pose = np.eye(4, dtype=np.float32)
    lm_pnp_pose[:3, :3] = R_res
    lm_pnp_pose[:3, 3] = lm_pnp_pose_vec[3:].ravel()

    # measure accuracy
    gt_pose = gt_Tcws.squeeze(0).detach().cpu().numpy()

    R_acc = rel_rot_angle(lm_pnp_pose, gt_pose)
    t_acc = rel_distance(lm_pnp_pose, gt_pose)

    #     ransc_inlier = None
    return R_acc, t_acc, lm_pnp_pose, inlier_map


def recover_original_scene_coordinates_all_level(query_X_ws, rand_R, scene_center):
    query_X_3d_ws = []
    for query_X_w in query_X_ws:
        query_X_3d_w = recover_original_scene_coordinates(query_X_w, rand_R, scene_center).permute(0, 3, 1, 2)
        query_X_3d_ws.append(query_X_3d_w)
    return query_X_3d_ws