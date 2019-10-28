import os
import sys
import cv2
import numpy as np
import torch
import torch.nn
from core_io.depth_io import load_depth_from_png
import matplotlib.pyplot as plt
from frame_seq_data import FrameSeqData
import core_3dv.camera_operator_gpu as cam_opt
# from visualizer.visualizer_3d import Visualizer
from evaluator.basic_metric import rel_rot_angle, rel_distance


def solve_pnp(K: torch.Tensor, x_2d: torch.Tensor, X_3d_w: torch.Tensor, reproj_thres=2.0):
    """
    Solve PnP problem with OpenCV lib
    :param K: camera intrinsic matrix, dim: (N, 3x3) or (3, 3)
    :param x_2d: 2D coordinates, dim: (N, H, W, 2), (H, W, 2),
    :param X_3d_w: 3D world coordinates, dim: (N, H, W, 2), (H, W, 3)
    :return:
    """

    keep_dim_n = False
    if K.dim() == 2:
        keep_dim_n = True
        K = K.unsqueeze(0)
        x_2d = x_2d.unsqueeze(0)
        X_3d_w = X_3d_w.unsqueeze(0)

    N, H, W = x_2d.shape[:3]
    K = K.detach().cpu().numpy()
    x_2d = x_2d.detach().cpu().numpy()
    X_3d_w = X_3d_w.view(N, -1, 3).detach().cpu().numpy()

    poses = []
    x_2d = x_2d[0].reshape(1, H*W, 2)
    dist = np.zeros(4)
    for n in range(N):
        k = K[n]
        X_3d = X_3d_w[n].reshape(1, H*W, 3)
        _, R_res, t_res, _ = cv2.solvePnPRansac(X_3d, x_2d, k, dist, reprojectionError=reproj_thres)
        R_res, _ = cv2.Rodrigues(R_res)

        pnp_pose = np.eye(4, dtype=np.float32)
        pnp_pose[:3, :3] = R_res
        pnp_pose[:3, 3] = t_res.ravel()
        poses.append(pnp_pose)

    poses = torch.cat([torch.from_numpy(pose) for pose in poses])

    if keep_dim_n is True:
        poses.squeeze(0)

    return poses

if __name__ == '__main__':
    vis = Visualizer()

    seq_dir = '/home/ziqianb/Desktop/datasets/tgz_target/'
    seq = FrameSeqData(os.path.join(seq_dir, 'rgbd_dataset_freiburg1_desk', 'seq.json'))
    frame_a = seq.frames[0]
    frame_b = seq.frames[12]

    depth_a = load_depth_from_png(os.path.join(seq_dir, seq.get_depth_name(frame_a)), div_factor=5000)
    depth_b = load_depth_from_png(os.path.join(seq_dir, seq.get_depth_name(frame_b)), div_factor=5000)
    img_a = cv2.imread(os.path.join(seq_dir, seq.get_image_name(frame_a))).astype(np.float32) / 255.0
    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    img_b = cv2.imread(os.path.join(seq_dir, seq.get_image_name(frame_b))).astype(np.float32) / 255.0
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

    depth_a = torch.from_numpy(depth_a).unsqueeze(0)
    K = torch.from_numpy(seq.get_K_mat(frame_a)).unsqueeze(0)
    H, W = depth_a.shape[1], depth_a.shape[2]
    Twc_a = torch.from_numpy(seq.get_Twc(frame_a)).unsqueeze(0)
    R_a, t_a = cam_opt.Rt(Twc_a)

    x_2d = cam_opt.x_2d_coords(H, W, n=1)
    X_3d_a = cam_opt.pi_inv(K=K, d=depth_a, x=x_2d)
    X_3d_a = cam_opt.transpose(R_a, t_a, X_3d_a)

    """ PnP By OpenCV
    """
    # X_3d_a = X_3d_a.view(H, W, 3).cpu().numpy()
    # x_2d = x_2d.view(H, W, 2).cpu().numpy()
    # K = K.view(3, 3).cpu().numpy()
    # dist = np.zeros(4)
    # _, R_res, t_res, _ = cv2.solvePnPRansac(X_3d_a.reshape(1, H*W, 3), x_2d.reshape(1, H*W, 2), K, dist)
    # R_res, _ = cv2.Rodrigues(R_res)

    pnp_pose = solve_pnp(K, x_2d, X_3d_a)
    pnp_pose = pnp_pose.cpu().numpy()

    gt_pose = seq.get_Tcw(frame_a)
    rel_trans = rel_distance(gt_pose, pnp_pose)
    rel_angle = rel_rot_angle(gt_pose, pnp_pose)
    print(rel_trans ,rel_angle)

    vis.add_frame_pose(pnp_pose[:3, :3], pnp_pose[:3, 3])
    X_3d_a = X_3d_a.view(H, W, 3).cpu().numpy()
    vis.set_point_cloud(X_3d_a.reshape((H * W, 3)), colors=img_a.reshape((H * W, 3)))
    vis.show()

