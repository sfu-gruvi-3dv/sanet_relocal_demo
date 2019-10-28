import os
import torch
import numpy as np
import cv2
from relocal.point_feat import extract_points, extract_points_validpool
from visualizer.visualizer_3d import Visualizer
from frame_seq_data import FrameSeqData
from core_io.depth_io import load_depth_from_png
import core_3dv.camera_operator_gpu as cam_opt
from libs.pycbf_filter.depth_fill import fill_depth_cross_bf
import matplotlib.pyplot as plt

vis = Visualizer()

valid_set_dir = '/home/ziqianb/Desktop/datasets/tgz_target/'
valid_seq_name = 'rgbd_dataset_freiburg1_desk'

seq = FrameSeqData(os.path.join(valid_set_dir, valid_seq_name, 'seq.json'))

frame_a = seq.frames[5]
frame_b = seq.frames[20]

Tcw_a = seq.get_Tcw(frame_a)
Tcw_b = seq.get_Tcw(frame_b)
K = seq.get_K_mat(frame_a)

img_a = cv2.imread(os.path.join(valid_set_dir, seq.get_image_name(frame_a))).astype(np.float32) / 255.0
img_b = cv2.imread(os.path.join(valid_set_dir, seq.get_image_name(frame_b))).astype(np.float32) / 255.0
depth_a = load_depth_from_png(os.path.join(valid_set_dir, seq.get_depth_name(frame_a)), div_factor=5000.0)
depth_b = load_depth_from_png(os.path.join(valid_set_dir, seq.get_depth_name(frame_b)), div_factor=5000.0)
H, W = img_a.shape[:2]

# plt.imshow(depth_a)
# plt.show()


depth_a = fill_depth_cross_bf(img_a, depth_a)
# plt.imshow(depth_a)
# plt.show()

Tcw_a = torch.from_numpy(Tcw_a).cuda().unsqueeze(0)
Tcw_b = torch.from_numpy(Tcw_b).cuda().unsqueeze(0)
K = torch.from_numpy(K).cuda().unsqueeze(0)
# img_a = torch.from_numpy(img_a).cuda().unsqueeze(0)
img_b = torch.from_numpy(img_b).cuda().unsqueeze(0)
depth_a = torch.from_numpy(depth_a).cuda().view(H, W).unsqueeze(0)
depth_b = torch.from_numpy(depth_b).cuda().view(H, W).unsqueeze(0)
valid_a_mask = (depth_a > 1e-5).expand(1, 3, H, W).float()

x_2d = cam_opt.x_2d_coords(H, W, n=1).to(depth_a.device)
X_3d_a = cam_opt.pi_inv(K, x_2d, depth_a)
R, t = cam_opt.Rt(Tcw_a)
X_3d = cam_opt.transpose(R, t, X_3d_a)


def keyPressEvent(obj, event):
    global img_a, X_3d, H, W, valid_a_mask
    key = obj.GetKeySym()

    if key == '1':
        X_3d_in = X_3d.permute(0, 3, 1, 2)
        conv_feat = torch.rand(1, 512, H // 2, W // 2)
        sub_X_3d, conv_feat = extract_points_validpool(X_3d_in, valid_a_mask, conv_feat, pool_kernel_size=2)

        img_t = cv2.resize(img_a, (320, 240))
        img_t = torch.from_numpy(img_t).cuda().unsqueeze(0)

        sub_X_3d = sub_X_3d.view((H // 2) * (W // 2), 3)
        x_color = img_t.view((H // 2) * (W // 2), 3)

        vis.set_point_cloud(points=sub_X_3d.detach().cpu().numpy(),
                            colors=x_color.detach().cpu().numpy())
    elif key == '2':
        X_3d_in = X_3d.permute(0, 3, 1, 2)
        conv_feat = torch.rand(1, 512, H // 16, W // 16)
        sub_X_3d, conv_feat = extract_points_validpool(X_3d_in, valid_a_mask, conv_feat, pool_kernel_size=16)

        img_t = cv2.resize(img_a, (40, 30))
        img_t = torch.from_numpy(img_t).cuda().unsqueeze(0)

        sub_X_3d = sub_X_3d.view((H // 16) * (W // 16), 3)
        x_color = img_t.view((H // 16) * (W // 16), 3)

        vis.set_point_cloud(points=sub_X_3d.detach().cpu().numpy(),
                            colors=x_color.detach().cpu().numpy())
    else:
        X_3d_in = X_3d.view(H*W, 3)

        img_t = torch.from_numpy(img_a).cuda().unsqueeze(0)
        x_color = img_t.view(H*W, 3)

        vis.set_point_cloud(points=X_3d_in.detach().cpu().numpy(),
                            colors=x_color.detach().cpu().numpy())

    return

vis.bind_keyboard_event(keyPressEvent)
vis.show()
# print(X_3d.shape)

