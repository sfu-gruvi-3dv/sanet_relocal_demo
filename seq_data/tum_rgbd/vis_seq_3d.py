import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from frame_seq_data import FrameSeqData, K_from_frame
import core_3dv.camera_operator as cam_opt
import torch
import banet_track.ba_module as module
from core_io.depth_io import load_depth_from_png
import core_math.transfom as trans
from visualizer.visualizer_2d import show_multiple_img
from visualizer.visualizer_3d import Visualizer
from seq_data.sun3d.read_util import read_sun3d_depth
from img_proc.img_dim import crop_by_intrinsic
from seq_data.plot_seq_2d import plot_frames_seq_2d
from vo_core.track_preprocess import convert_rel_vo

''' Configuration ------------------------------------------------------------------------------------------------------
'''
base_dir = '/home/ziqianb/Desktop/tgz'

seq_name = 'rgbd_dataset_freiburg1_xyz'

frames = FrameSeqData(os.path.join(base_dir, seq_name, 'seq.json'))
refer_T = frames.get_Tcw(frames.frames[0])
convert_rel_vo(frames, refer_T)

in_intrinsic = np.asarray([0.88 * 512, 1.17 * 384, 256.0, 192.0], dtype=np.float32)
in_K = cam_opt.K_from_intrinsic(in_intrinsic)

''' Scripts ------------------------------------------------------------------------------------------------------------
'''
plot_frames_seq_2d(frames, show_view_direction=True)
plt.show()

vis = Visualizer()
frame_idx = 292
x_2d = cam_opt.x_2d_coords(int(in_intrinsic[3]*2), int(in_intrinsic[2]*2))
# x_2d = cam_opt.x_2d_coords(480, 640)

def keyPressEvent(obj, event):
    global frame_idx
    global refer_T
    key = obj.GetKeySym()
    if key == 'Right':
        # vis.clear_frame_poses()
        if frame_idx > 305:
            return

        cur_frame = frames.frames[frame_idx]
        cur_Tcw = cur_frame['extrinsic_Tcw']
        cur_name = cur_frame['file_name']
        cur_depth_name = cur_frame['depth_file_name']
        print(cur_name)

        K = K_from_frame(cur_frame)

        # Read image
        cur_img = cv2.imread(os.path.join(base_dir, cur_name)).astype(np.float32) / 255.0
        cur_depth = load_depth_from_png(os.path.join(base_dir, cur_depth_name), div_factor=5000)

        # Crop with new intrinsic
        cur_img = crop_by_intrinsic(cur_img, K, in_K, interp_method='nearest')
        # next_img = crop_by_intrinsic(next_img, K, in_K)
        cur_depth = crop_by_intrinsic(cur_depth, K, in_K, interp_method='nearest')
        h, w, c = cur_img.shape

        # rel_T = cam_opt.relateive_pose(cur_Tcw[:3, :3], cur_Tcw[:3, 3], next_Tcw[:3, :3], next_Tcw[:3, 3])
        X_3d = cam_opt.pi_inv(K, x_2d.reshape((h*w, 2)), cur_depth.reshape((h*w, 1)))
        cur_Twc = cam_opt.camera_pose_inv(cur_Tcw[:3, :3], cur_Tcw[:3, 3])
        X_3d = cam_opt.transpose(cur_Twc[:3, :3], cur_Twc[:3, 3], X_3d)

        vis.set_point_cloud(X_3d, cur_img.reshape((h*w, 3)))
        vis.add_frame_pose(cur_Tcw[:3, :3], cur_Tcw[:3, 3], camera_obj_scale=0.01)

        frame_idx += 1

    return

vis.bind_keyboard_event(keyPressEvent)
vis.show()
