import os

import numpy as np
import torch
import core_3dv.camera_operator as camera_op
import img_proc.img_dim
import cv2
from img_proc.basic_proc import gradient
from frame_seq_data import FrameSeqData, K_from_frame
from core_io.depth_io import load_depth_from_pgm
import matplotlib.pyplot as plt
import core_math.transfom as trans
from visualizer.visualizer_3d import Visualizer

from banet_track.ba_module import J_camera_pose, batched_interp2d, batched_index_select, batched_gradient



def x_2d_coords(n, h, w):
    x_2d = np.zeros((n, h, w, 2), dtype=np.float32)
    for y in range(0, h):
        x_2d[:, y, :, 1] = y
    for x in range(0, w):
        x_2d[:, :, x, 0] = x
    return torch.Tensor(x_2d)


def photometric_error(I_a, sel_pt_idx, I_b, x_b_2d):

    N = I_a.shape[0]  # number of batches
    M = sel_pt_idx.shape[1]  # number of samples
    C = I_a.shape[1]  # number of channels
    H = I_a.shape[2]
    W = I_a.shape[3]

    # Wrap the image
    Ib_wrap = batched_interp2d(I_b, x_b_2d)

    # Intensity error
    e = I_a - Ib_wrap

    # select choosen indecs
    e = e.view(N, C, H * W)
    e = batched_index_select(e, 2, sel_pt_idx)

    return e


def Jac(X_3d, grad, fx, fy):
    """
    [TESTED] with numeric, when transformation is Identity Mat, other transformation has problem.
    Compute the Jacobin of Camera pose
    :param X_3d: 3D Points Position, dim: (N, M, 3), N is the batch size, M is the number sampled points
    :param grad: feature gradient, dim: (N, C, M, 2)
    :param fx: focal length on x dim (float32)
    :param fy: focal length on y dim (float32)
    :return: Jacobin Mat Tensor with Dim (N, M*2, 6) where the (M*2, 6) represent the Jacobin matrix and N is the batches
    """
    N = X_3d.shape[0]  # number of batches
    M = X_3d.shape[1]  # number of samples
    C = grad.shape[1]  # number of channels

    # compute camera Jacobin
    J_camera = J_camera_pose(X_3d, fx, fy).view(N, 1, M, 2, 6).expand(N, C, M, 2, 6).view(N * C * M, 2, 6)

    # compute final Jacobin, left multiply feature gradient
    grad = grad.view(N * C * M, 1, 2)
    J = torch.bmm(grad, J_camera).view(N, C * M, 6)

    return -J
