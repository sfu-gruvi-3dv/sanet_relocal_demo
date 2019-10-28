import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import frame_seq_data
import seq_data.plot_seq_2d as plt_seq
import seq_data.random_sel_frames
from visualizer.visualizer_2d import show_multiple_img

''' Configuration ------------------------------------------------------------------------------------------------------
'''
# SUN3D Base dir
base_dir = '/home/luwei/mnt/Tango/ziqianb/SUN3D/'

seq_name = 'brown_bm_2/brown_bm_2'

# Select a sequences
ori_seq_json_path = os.path.join(base_dir, seq_name, 'seq.json')

# Toggle to show 2D seq map instead of image sequences
show_2d_path = True

# Load the original frame and random sample subset
ori_seq = frame_seq_data.FrameSeqData(ori_seq_json_path)
sub_seq_list = seq_data.random_sel_frames.rand_sel_subseq_sun3d(scene_frames=ori_seq,
                                                                trans_thres_range=0.15,
                                                                frames_per_subseq_num=10,
                                                                frames_range=(0.00, 0.8),
                                                                max_subseq_num=30,
                                                                interval_thres=2)

''' Scripts  -----------------------------------------------------------------------------------------------------------
'''
if show_2d_path:
    plt.figure()
    ax = plt.gca()
    plt_seq.plot_frames_seq_2d(ori_seq, ax, legend='all')
    for sub_seq in sub_seq_list:
        plt_seq.plot_frames_seq_2d(sub_seq, ax, point_style='x-')
    plt.show()
else:
    for seq in sub_seq_list:
        img_list = []
        for frame in seq.frames:
            cur_name = frame['file_name']
            cur_frame_idx = frame['id']
            cur_img = cv2.imread(os.path.join(base_dir, cur_name)).astype(np.float32) / 255.0
            img_list.append({'img': cur_img, "title": cur_frame_idx})
        show_multiple_img(img_list)
