import os
import sys
import numpy as np
import cv2
import math
import h5py
import json
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from core_math.transfom import quaternion_matrix, rotation_from_matrix, quaternion_from_matrix
from banet_track.ba_module import batched_select_gradient_pixels, se3_exp, batched_relative_pose, x_2d_coords_torch, batched_pi_inv, batched_pi, batched_quaternion2rot
from seq_data.seq_preprocess import add_gaussian_noise, add_drift_noise

# sys.path.extend(['/opt/eigency', '/opt/PySophus'])
# from sophus import *

def relative_angle(q1, q2):
    # Has numeric stability issue, the minimal degree may be 0.0396
    t = 2.0*torch.sum(q1 * q2, dim=-1)**2 - 1.0
    t = torch.clamp(t, -1.0, 1.0)
    t = torch.acos(t)
    t = 180 * t / np.pi
    return t

def ba_tracknet_preprocess(frames_dict, num_pyramid=3, M=2000):
    """
    preprocess ImagePairDataset for ba_tracknet, generate selected gradient pixel indices
    :param frames_dict: dict returned from dataloader
    :param num_pyramid: number of feature map pyramids used in ba_tracknet
    :param M: the maximum number of pixels we want to select in one feature map
    :return: all variables used in ba_tracknet
                    I_a: Image of frame A, dim: (N, C, H, W)
                    d_a: Depth of frame A, dim: (N, 1, H, W)
                    sel_a_indices: selected point index on num_pyramid-level, dim: (N, num_pyramid, M)
                    K: intrinsic matrix at level 0: dim: (N, 3, 3)
                    I_b: Image of frame B, dim: (N, C, H, W)
                    q_gt : Groundtruth of quaternion, dim: (N, 4)
                    t_gt: Groundtruth of translation, dim: (N, 3)
                    se3_gt: Groundtruth of se3, dim: (N, 6)
    """
    N, C, H, W = frames_dict['frame0']['img'].shape
    I_a = frames_dict['frame0']['img']
    # I_a = F.interpolate(I_a, scale_factor=0.5)
    gray_img_tensor = I_a[:, 0:1, :, :] * 0.299 + I_a[:, 1:2, :, :] * 0.587 + I_a[:, 2:3, :, :] * 0.114
    d_a = frames_dict['frame0']['depth']
    # d_a = F.interpolate(d_a, scale_factor=0.5)
    K = frames_dict['frame0']['K']
    I_b = frames_dict['frame1']['img']
    # I_b = F.interpolate(I_b, scale_factor=0.5)
    Tcw = frames_dict['frame1']['Tcw']
    sel_a_indices = batched_select_gradient_pixels(gray_img_tensor, d_a, I_b, K, Tcw[:, :, :3], Tcw[:, :, 3], grad_thres=15.0 / 255.0,
                                                   num_pyramid=num_pyramid, num_gradient_pixels=M, visualize=False)
    Tcw_np = Tcw.numpy()
    # print(Tcw[0])
    q_gt = torch.empty((N, 4), dtype=torch.float, device=torch.device('cpu'))
    t_gt = torch.empty((N, 3), dtype=torch.float, device=torch.device('cpu'))
    se3_gt = torch.empty((N, 6), dtype=torch.float, device=torch.device('cpu'))
    for i in range(N):
        R_mat = np.eye(4)
        R_mat[:3, :3] = Tcw_np[i, :3, :3]
        q = quaternion_from_matrix(R_mat)
        q_gt[i, :] = torch.Tensor(q)
        t_gt[i, :] = torch.Tensor(Tcw_np[i, :3, 3])

        T_mat = R_mat
        T_mat[:3, 3] = Tcw_np[i, :3, 3]
        # T = SE3(T_mat)
        T = None
        t = T.log().ravel()
        se3_gt[i, :3] = torch.Tensor(t[3:])
        se3_gt[i, 3:] = torch.Tensor(t[:3])
    # print(quaternion_matrix(q_gt[0].numpy()), t_gt[0].numpy())
    # R, t = se3_exp(se3_gt)
    # print(R[0].numpy(), t[0].numpy())

    return I_a, d_a, sel_a_indices, K, I_b, q_gt, t_gt, se3_gt, Tcw


def lstm_preprocess(seq_dict, num_pyramid=3, M=2000, add_noise_func=add_drift_noise, rot_noise_deg=10.0, displacement_dist_std=0.1):
    """
    preprocess SUN3DSeqDataset for LSTMNet, generate selected gradient pixel indices
    :param seq_dict: dict returned from dataloader
    :param num_pyramid: number of feature map pyramids used in ba_tracknet and lstm_net
    :param M: the maximum number of pixels we want to select in one feature map
    :return: all variables used in LSTMNet
                    I: frame images of the sequence, (N, F, C, H, W)
                    d: depth maps of the sequence, (N, F, 1, H, W)
                    sel_indices: selected indices of pixels of each frame, (N, F, num_pyramid, M)
                    K: intrinsic matrix at level 0: dim: (N, F, 3, 3)
                    T: noised pose, (N, F, 4, 4)
                    T_gt: ground truth pose, (N, F. 4, 4)
    """
    I = seq_dict['img']
    d = seq_dict['depth']
    K = seq_dict['K']
    Tcw = seq_dict['Tcw']
    Tcw_np = Tcw.numpy()
    gray_img_tensor = I[:, :, 0:1, :, :] * 0.299 + I[:, :, 1:2, :, :] * 0.587 + I[:, :, 2:3, :, :] * 0.114

    N, L, C, H, W = I.shape

    # add noise to camera pose
    T_gt = np.eye(4, dtype=np.float32).reshape((1, 4, 4)).repeat(N * L, axis=0).reshape((N, L, 4, 4))
    T_gt[:, :, :3, :] = Tcw_np
    T_np = np.eye(4, dtype=np.float32).reshape((1, 4, 4)).repeat(N * L, axis=0).reshape((N, L, 4, 4))
    for i in range(N):
        noise_T, rand_std_radius = add_noise_func(Tcw_np[i], rot_noise_deg=rot_noise_deg, displacement_dist_std=displacement_dist_std)
        T_np[i, :, :3, :] = noise_T
    T_gt = torch.from_numpy(T_gt)
    T = torch.from_numpy(T_np)

    # convert rotation to quaternion
    q = np.empty((N, L, 4), dtype=np.float32)
    t = np.empty((N, L, 3), dtype=np.float32)
    q_gt = np.empty((N, L, 4), dtype=np.float32)
    t_gt = np.empty((N, L, 3), dtype=np.float32)
    for i in range(N):
        for j in range(L):
            q_gt[i, j, :] = quaternion_from_matrix(Tcw_np[i, j, :3, :3].copy())
            t_gt[i, j, :] = Tcw_np[i, j, :3, 3]
            q[i, j, :] = quaternion_from_matrix(T_np[i, j, :3, :3].copy())
            t[i, j, :] = T_np[i, j, :3, 3]
    q_gt = torch.from_numpy(q_gt)
    t_gt = torch.from_numpy(t_gt)
    tq_gt = torch.cat([t_gt, q_gt], dim=2)
    q = torch.from_numpy(q)
    t = torch.from_numpy(t)
    tq = torch.cat([t, q], dim=2)

    # test
    # Compute Accuracy, noise level
    # init_q_accu = 0.0
    # init_t_accu = 0.0
    # for i in range(1, tq.shape[0]):
    #     cur_gt_abs_tq = q_module.invert_pose_quaternion(tq_gt[i, :, :])
    #     cur_init_abs_tq = q_module.invert_pose_quaternion(tq[i, :, :])
    #     init_q_accu = torch.mean(relative_angle(cur_init_abs_tq[:, 3:], cur_gt_abs_tq[:, 3:]))
    #     init_t_accu = torch.sqrt(F.mse_loss(cur_init_abs_tq[:, :3], cur_gt_abs_tq[:, :3]))
    #     print(init_q_accu)
    #     print(init_t_accu)

    # init_q_accu /= (tq.shape[0] - 1)
    # init_t_accu /= (tq.shape[0] - 1)

    # print(Tcw_np[0, 1])
    # print(T_gt[0, 1])
    # rec_R = batched_quaternion2rot(q.view(N * F, 4))
    # print(rec_R[0], T[0, 0, :3, :3])
    # print(t[0, 0], T[0, 0, :3, 3])
    # rec_R_gt = batched_quaternion2rot(q_gt.view(N * F, 4))
    # print(rec_R_gt[0], T_gt[0, 0, :3, :3])
    # print(t_gt[0, 0], T_gt[0, 0, :3, 3])

    # select pixels at gradient edge
    sel_indices = torch.empty(N, L, num_pyramid, M, dtype=torch.long)
    # for i in range(1, F):
    #     rel_T = batched_relative_pose(Tcw[:, i, :3, :3], Tcw[:, i, :3, 3], Tcw[:, i - 1, :3, :3], Tcw[:, i - 1, :3, 3])
    #     sel_indices[:, i, :, :] = batched_select_gradient_pixels(gray_img_tensor[:, i, :, :, :], d[:, i, :, :, :],
    #                                                              I[:, i - 1, :, :, :], K[:, i, :, :], rel_T[:, :, :3], rel_T[:, :, 3],
    #                                                              grad_thres=15.0 / 255.0,
    #                                                              num_pyramid=num_pyramid, num_gradient_pixels=M, visualize=False)
    # rel_T = batched_relative_pose(Tcw[:, 0, :3, :3], Tcw[:, 0, :3, 3], Tcw[:, 1, :3, :3], Tcw[:, 1, :3, 3])
    # sel_indices[:, 0, :, :] = batched_select_gradient_pixels(gray_img_tensor[:, 0, :, :, :], d[:, 0, :, :, :],
    #                                                          I[:, 1, :, :, :], K[:, 0, :, :], rel_T[:, :, :3], rel_T[:, :, 3],
    #                                                          grad_thres=15.0 / 255.0,
    #                                                          num_pyramid=num_pyramid, num_gradient_pixels=M, visualize=False)
    return I.transpose(0, 1).contiguous(), d.transpose(0, 1).contiguous(), sel_indices.transpose(0, 1).contiguous(),\
           K.transpose(0, 1).contiguous(), T.transpose(0, 1).contiguous(), T_gt.transpose(0, 1).contiguous(),\
           tq.transpose(0, 1).contiguous(), tq_gt.transpose(0, 1).contiguous()
