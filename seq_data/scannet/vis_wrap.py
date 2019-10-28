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

seq_name = 'scene0105_02'

frames = FrameSeqData(os.path.join(base_dir, seq_name, 'seq.json'))


''' Script -------------------------------------------------------------------------------------------------------------
'''
x_2d = cam_opt.x_2d_coords(240, 320)
for frame_idx in range(10, len(frames), 5):

    cur_frame = frames.frames[frame_idx]
    cur_Tcw = cur_frame['extrinsic_Tcw']
    cur_name = cur_frame['file_name']
    cur_depth_name = cur_frame['depth_file_name']

    next_frame = frames.frames[frame_idx + 10]
    next_Tcw = next_frame['extrinsic_Tcw']
    next_name = next_frame['file_name']

    K = K_from_frame(cur_frame)

    # Read image
    cur_img = cv2.imread(os.path.join(base_dir, cur_name)).astype(np.float32) / 255.0
    next_img = cv2.imread(os.path.join(base_dir, next_name)).astype(np.float32) / 255.0
    cur_depth = load_depth_from_png(os.path.join(base_dir, cur_depth_name))
    h, w, c = cur_img.shape

    rel_T = cam_opt.relateive_pose(cur_Tcw[:3, :3], cur_Tcw[:3, 3], next_Tcw[:3, :3], next_Tcw[:3, 3])

    # Translation
    Cb = cam_opt.camera_center_from_Tcw(rel_T[:3, :3], rel_T[:3, 3])
    baseline = np.linalg.norm(Cb)

    # View angle
    q = trans.quaternion_from_matrix(rel_T)
    R = trans.quaternion_matrix(q)
    rel_rad, rel_axis, _ = trans.rotation_from_matrix(R)
    rel_deg = np.rad2deg(rel_rad)

    next2cur, _ = cam_opt.wrapping(cur_img, next_img, cur_depth, K, rel_T[:3, :3], rel_T[:3, 3])
    show_multiple_img([{'img': cur_img, 'title': 'a'},
                       {'img': next2cur, 'title': 'wrap_b2a'},
                       {'img': next_img, 'title': 'b'},
                       {'img': cur_depth.reshape((h, w)), 'title': 'depth', 'cmap':'jet'}],
                        title='rel_deg: %f, rel_trans: %f' % (rel_deg, baseline))

    break