import numpy as np
import torch
import cv2
import os
from frame_seq_data import FrameSeqData
from seq_data.sun3d.read_util import read_sun3d_depth
import core_3dv.camera_operator as cam_opt
from numba import cuda

@cuda.jit
def check_consistency_kernel(img_a, img_b, depth_a, depth_b, K):
    h = img_a.shape[0]
    w = img_a.shape[1]




def filter_seq3(seq_list: FrameSeqData, base_dir):

    for seq in seq_list.frames:

        pre_frame = seq[0]
        center_frame = seq[1]
        next_frame = seq[2]

        pre_Tcw = seq_list.get_Tcw(pre_frame)
        center_Tcw = seq_list.get_Tcw(center_frame)
        next_Tcw = seq_list.get_Tcw(next_frame)

        K_mat = seq_list.get_K_mat(center_frame)

        # Read Image
        pre_img_name = seq_list.get_image_name(pre_frame)
        center_img_name = seq_list.get_image_name(center_frame)
        next_img_name = seq_list.get_image_name(pre_frame)
        pre_img = cv2.imread(os.path.join(base_dir, pre_img_name)).astype(np.float32) / 255.0
        center_img = cv2.imread(os.path.join(base_dir, center_img_name)).astype(np.float32) / 255.0
        next_img = cv2.imread(os.path.join(base_dir, next_img_name)).astype(np.float32) / 255.0

        # Read depth
        pre_depth_name = seq_list.get_depth_name(pre_frame)
        center_depth_name = seq_list.get_depth_name(center_frame)
        next_depth_name = seq_list.get_depth_name(next_frame)
        pre_depth = read_sun3d_depth(pre_depth_name)
        center_depth = read_sun3d_depth(center_depth_name)
        next_depth = read_sun3d_depth(next_depth_name)

        # cam_opt.dense_corres_a2b(center_depth, K_mat, center_Tcw, )



