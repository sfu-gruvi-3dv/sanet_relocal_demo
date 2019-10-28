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

''' Configuration ------------------------------------------------------------------------------------------------------
'''
base_dir = '/mnt/Exp_3/scannet'

seq_name = 'scene0131_02'

frames = FrameSeqData(os.path.join(base_dir, seq_name, 'seq.json'))

''' Scripts ------------------------------------------------------------------------------------------------------------
'''
vis = Visualizer()

frame_idx = 0
x_2d = cam_opt.x_2d_coords(480, 640)

def keyPressEvent(obj, event):
    global frame_idx
    key = obj.GetKeySym()
    if key == 'Right':
        cur_frame = frames.frames[frame_idx]
        cur_Tcw = cur_frame['extrinsic_Tcw']
        cur_name = cur_frame['file_name']
        cur_depth_name = cur_frame['depth_file_name']

        next_frame = frames.frames[frame_idx + 1]
        next_Tcw = next_frame['extrinsic_Tcw']
        next_name = next_frame['file_name']

        K = K_from_frame(cur_frame)

        # Read image
        cur_img = cv2.imread(os.path.join(base_dir, cur_name)).astype(np.float32) / 255.0
        next_img = cv2.imread(os.path.join(base_dir, next_name)).astype(np.float32) / 255.0
        cur_depth = load_depth_from_png(os.path.join(base_dir, cur_depth_name))
        h, w, c = cur_img.shape

        rel_T = cam_opt.relateive_pose(cur_Tcw[:3, :3], cur_Tcw[:3, 3], next_Tcw[:3, :3], next_Tcw[:3, 3])
        X_3d = cam_opt.pi_inv(K, x_2d.reshape((h*w, 2)), cur_depth.reshape((h*w, 1)))
        cur_Twc = cam_opt.camera_pose_inv(cur_Tcw[:3, :3], cur_Tcw[:3, 3])
        X_3d = cam_opt.transpose(cur_Twc[:3, :3], cur_Twc[:3, 3], X_3d)

        vis.set_point_cloud(X_3d, cur_img.reshape((h*w, 3)))
        vis.add_frame_pose(cur_Tcw[:3, :3], cur_Tcw[:3, 3])

        frame_idx += 20

    return

vis.bind_keyboard_event(keyPressEvent)
vis.show()
