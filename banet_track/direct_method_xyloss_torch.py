import os

import numpy as np
import core_3dv.camera_operator as camera_op
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

from banet_track.ba_module import J_camera_pose, batched_interp2d, batched_index_select, batched_gradient, batched_pi, \
    batched_pi_inv, se3_exp, batched_transpose, batched_x_2d_normalize
from banet_track.ba_optimizer import gauss_newtown_update, levenberg_marquardt_update


def x_2d_coords_torch(n, h, w):
    x_2d = np.zeros((n, h, w, 2), dtype=np.float32)
    for y in range(0, h):
        x_2d[:, y, :, 1] = y
    for x in range(0, w):
        x_2d[:, :, x, 0] = x
    return torch.from_numpy(x_2d).cuda()


def select_gradient_pixels(img, depth, grad_thres=0.1, depth_thres=1e-5, visualize=False):
    h, w = img.shape[0], img.shape[1]
    grad = gradient(img) / 2.0
    grad_norm = np.linalg.norm(grad, axis=2)
    mask = np.logical_and(grad_norm > grad_thres, depth > depth_thres)
    sel_index = np.asarray(np.where(mask.reshape(h*w)), dtype=np.int)

    # Visualize
    if visualize:
        selected_mask = np.zeros((h*w), dtype=np.float32)
        selected_mask[sel_index] = 1.0

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(img, cmap='gray')
        axes[1].imshow(selected_mask.reshape(h, w), cmap='gray')
        plt.show()

    np.random.shuffle(sel_index)
    return sel_index.ravel()

def select_nonzero_pixels(img, depth, depth_thres=1e-5, visualize=False):
    h, w = img.shape[0], img.shape[1]
    mask = np.logical_and(depth > depth_thres, True)
    sel_index = np.asarray(np.where(mask.reshape(h*w)), dtype=np.int)

    # Visualize
    if visualize:
        selected_mask = np.zeros((h*w), dtype=np.float32)
        selected_mask[sel_index] = 1.0

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(img, cmap='gray')
        axes[1].imshow(selected_mask.reshape(h, w), cmap='gray')
        plt.show()

    np.random.shuffle(sel_index)
    return sel_index.ravel()

''' Torch
'''
def compute_pose(I_a, d_a, sel_a_idx, I_b, K, alpha, T_gt, opt_max_itr=100, opt_eps=1e-5):

    # Debug assert
    assert sel_a_indices.dtype == torch.int64
    assert I_a.dtype == torch.float32
    assert I_b.dtype == torch.float32
    assert d_a.dtype == torch.float32

    # Dimension
    N, C, H, W = I_a.shape

    # Pre-processing
    # sel_a_idx = select_gradient_pixels(I_a, d_a, threshold=50.0)[: 2000]
    M = sel_a_idx.shape[1]
    I_b_grad = batched_gradient(I_b)              # dim: (N, 2*C, H, W), (N, 0:C, H, W) = dI/dx, (N, C:2C, H, W) = dI/dy

    assert H == d_a.shape[2]
    assert W == d_a.shape[3]

    # se(3) vector init
    lambda_w = 0.2*torch.ones(N, 6)
    d_a = d_a.view((N, H*W, 1))

    # Points' 3D Position at Frame a
    x_a_2d = x_2d_coords_torch(N, H, W).view(N, H*W, 2)
    X_a_3d = batched_pi_inv(K, x_a_2d, d_a)
    X_a_3d_sel = batched_index_select(X_a_3d, 1, sel_a_idx)

    # groundtruth wrap
    alpha_gt = torch.tensor([0.5, 0.5, 0.5, 0.0, 0.0, 0.0]).repeat(N).view((N, 6))
    R_gt, _ = se3_exp(alpha_gt)
    I = torch.eye(3).view(1, 3, 3).expand(N, 3, 3).cuda()
    zeros = torch.zeros_like(T_gt[:, :, 3]).cuda()
    random_t = torch.zeros_like(T_gt[:, :, 3]).normal_(std=0.001)
    #print('random_t:', random_t)
    X_b_3d_gt = batched_transpose(R_gt, zeros, X_a_3d_sel)
    x_b_2d_gt, _ = batched_pi(K, X_b_3d_gt)

    for itr in range(0, opt_max_itr):

        R, t = se3_exp(alpha)

        X_b_3d = batched_transpose(R, t, X_a_3d_sel)
        x_b_2d, _ = batched_pi(K, X_b_3d)

        # Residual error
        e = (x_b_2d_gt - x_b_2d).view(N, M * 2)                                                     # (N, H*W*2)

        # Compute Jacobin Mat.
        # Jacobi of Camera Pose: delta_u / delta_alpha
        J = - J_camera_pose(X_a_3d_sel, K).view(N, M * 2, 6)  # (N*M, 2, 6)


        # x_b_2d = batched_x_2d_normalize(H, W, x_b_2d).view(N, H, W, 2)                              # (N, H, W, 2)
        #
        # # Wrap the image
        # I_b_wrap = batched_interp2d(I_b, x_b_2d)
        #
        # # Residual error
        # e = (I_a - I_b_wrap).view(N, C, H*W)                                                        # (N, C, H, W)
        # e = batched_index_select(e, 2, sel_a_idx)                                                   # (N, C, M)
        # e = e.transpose(1, 2).contiguous().view(N, M*C)                                             # (N, M, C)
        #
        # # Compute Jacobin Mat.
        # # Jacobi of Camera Pose: delta_u / delta_alpha
        # du_d_alpha = J_camera_pose(X_a_3d_sel, K).view(N * M, 2, 6)                                 # (N*M, 2, 6)
        #
        # # Jacobi of Image gradient: delta_I_b / delta_u
        # dI_du = batched_interp2d(I_b_grad, x_b_2d)                                                  # (N, 2*C, H, W)
        # dI_du = batched_index_select(dI_du.view(N, 2*C, H*W), 2, sel_a_idx)                         # (N, 2*C, M)
        # dI_du = torch.transpose(dI_du, 1, 2).contiguous().view(N * M, 2, C)                                      # (N*M, 2, C)
        # dI_du = torch.transpose(dI_du, 1, 2)                                                        # (N*M, C, 2)
        #
        # # J = -dI_b/du * du/d_alpha
        # J = -torch.bmm(dI_du, du_d_alpha).view(N, C*M, 6)

        # Compute the update parameters
        delta, delta_norm = gauss_newtown_update(J, e)                                              # (N, 6), (N, 1)
        max_norm = torch.max(delta_norm).item()
        if max_norm < opt_eps:
            print('break')
            break

        r_norm = torch.sum(e*e, dim=1) / M#2.0
        print('Itr:', itr, 'r_norm=', torch.sqrt(r_norm), "update_norm=", max_norm)

        # Update the delta
        alpha = alpha + delta

    return R, t

''' Pipeline -----------------------------------------------------------------------------------------------------------
'''
torch.set_default_tensor_type('torch.cuda.FloatTensor')

dataset_dir = '/home/ziqianb/Documents/data/RGBD-SLAM/rgbd_dataset_freiburg1_xyz/ares_output'
img_dir = os.path.join(dataset_dir, 'img')
depth_dir = os.path.join(dataset_dir, 'depth')
# vis = Visualizer(1024, 720)

# Load the frames
frames = FrameSeqData()
frames.load_json(os.path.join(dataset_dir, 'keyframe.json'))

# Test on frames
pairs = [(235, 237), (128, 131)]

I_a_set = []
d_a_set = []
I_b_set = []
K_set = []
sel_a_idx_set = []
T_gt_set = []
for pair in pairs:
    frame_a = frames.frames[pair[0]]
    file_name = frame_a['file_name']
    img = cv2.imread(os.path.join(img_dir, file_name + '.jpg')).astype(np.float32)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                    # Use for selecting pixels
    depth = load_depth_from_pgm(os.path.join(depth_dir, file_name + '.pgm'))
    depth[depth < 1e-5] = 1e-5
    sel_a_indices = select_nonzero_pixels(gray_img, depth, visualize=True)[:2000]

    img = img.transpose((2, 0, 1)) / 255.0
    pose = frame_a['extrinsic_Tcw']
    K = K_from_frame(frame_a)

    # Next frame
    frame_b = frames.frames[pair[1]]
    next_file_name = frame_b['file_name']
    next_img = cv2.imread(os.path.join(img_dir, next_file_name + '.jpg')).astype(np.float32)
    next_img = next_img.transpose((2, 0, 1)) / 255.0
    next_pose = frame_b['extrinsic_Tcw']

    T_gt = camera_op.relateive_pose(pose[:3, :3], pose[:3, 3], next_pose[:3, :3], next_pose[:3, 3])

    I_a_set.append(img)
    d_a_set.append(depth)
    I_b_set.append(next_img)
    K_set.append(K)
    sel_a_idx_set.append(sel_a_indices.reshape((1, 2000)))
    T_gt_set.append(T_gt)

    # Compute the pose
    # sel_a_indices = torch.from_numpy(sel_a_indices).cuda().view((1, -1)).long()

I_a_set = np.stack(I_a_set, axis=0)
d_a_set = np.stack(d_a_set, axis=0)
I_b_set = np.stack(I_b_set, axis=0)
K_set = np.stack(K_set, axis=0)
sel_a_idx_set = np.stack(sel_a_idx_set, axis=0)
T_gt_set = np.stack(T_gt_set, axis=0)

N = I_a_set.shape[0]
(C, H, W) = (I_a_set.shape[1:])

sel_a_indices = torch.from_numpy(sel_a_idx_set).cuda().view((N, -1)).long()

I_a = torch.from_numpy(I_a_set).cuda().view((N, C, H, W))
d_a = torch.from_numpy(d_a_set).cuda().view((N, 1, H, W))
I_b = torch.from_numpy(I_b_set).cuda().vieｗ((N, C, H, W))
K = torch.from_numpy(K_set).cuda().view((N, 3, 3))
T_gt = torch.from_numpy(T_gt_set).cuda().view((N, 3, 4))

alpha = torch.tensor([1e-4, 1e-4, 1e-4, 0.0, 0.0, 0.0]).repeat(N).view((N, 6))
R_, t_ = compute_pose(I_a, d_a, sel_a_indices, I_b, K, alpha, T_gt)
print(R_, t_)
print(T_gt[:, :, :3], T_gt[:, :, 3])



d_a = d_a.view((N, H*W, 1))

# Points' 3D Position at Frame a
x_a_2d = x_2d_coords_torch(N, H, W).view(N, H*W, 2)
X_a_3d = batched_pi_inv(K, x_a_2d, d_a)

# groundtruth wrap
X_b_3d_gt = batched_transpose(T_gt[:, :, :3], T_gt[:, :, 3], X_a_3d)
x_b_2d_gt, _ = batched_pi(K, X_b_3d_gt)

#
# """ Test on 235 to 237
# """
# frame = frames.frames[128]
# # Current frame
# file_name = frame['file_name']
# img = cv2.imread(os.path.join(img_dir, file_name + '.jpg')).astype(np.float32)
# (H, W, C) = img.shape
# img = img.transpose((2, 0, 1)) / 255.0
# depth = load_depth_from_pgm(os.path.join(depth_dir, file_name + '.pgm'))
# depth[depth < 1e-5] = 1e-5
# pose = frame['extrinsic_Tcw']
# K = K_from_frame(frame)
#
# # Next frame
# next_frame = frames.frames[131]
# next_file_name = next_frame['file_name']
# next_img = cv2.imread(os.path.join(img_dir, next_file_name + '.jpg')).astype(np.float32)
# next_img = next_img.transpose((2, 0, 1)) / 255.0
# next_pose = next_frame['extrinsic_Tcw']
# T_gt = camera_op.relateive_pose(pose[:3, :3], pose[:3, 3], next_pose[:3, :3], next_pose[:3, 3])
# R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]
#
# # Compute the pose
# sel_a_indices = np.load('sel_a_idx.npy')
# sel_a_indices = torch.from_numpy(sel_a_indices).cuda().view((1, -1)).long()
#
# I_a = torch.from_numpy(img).cuda().view((1, C, H, W))
# d_a = torch.from_numpy(depth).cuda().view((1, 1, H, W))
# I_b = torch.from_numpy(next_img).cuda().vieｗ((1, C, H, W))
# K = torch.from_numpy(K).cuda().view((1, 3, 3))
# alpha = torch.tensor([1e-6, 1e-6, 1e-6, 0.0, 0.0, 0.0]).repeat(1).view((1, 6))
# alpha.requires_grad_()
# R_, t_ = compute_pose(I_a, d_a, sel_a_indices, I_b, K, alpha)
# # torch.autograd.backward([R_, t_], [torch.ones(R_.shape), torch.ones(t_.shape)], retain_graph=True)
# # print(alpha.grad)
# print(R_, t_)
#

