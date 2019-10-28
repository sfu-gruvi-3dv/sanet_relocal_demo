from frame_seq_data import FrameSeqData
from core_3dv.camera_operator import camera_center_from_Tcw, relateive_pose, photometric_overlap, x_2d_coords
import core_math.transfom as trans
import copy
import numpy as np
import matplotlib.pyplot as plt
from seq_data.plot_seq_2d import plot_frames_seq_2d
import json
from core_io.depth_io import load_depth_from_png
from seq_data.sun3d.read_util import read_sun3d_depth
from evaluator.basic_metric import rel_rot_angle, rel_distance
import os
import cv2
import core_3dv.camera_operator as cam_opt

def rand_sel_subseq_sun3d(scene_frames,
                          max_subseq_num,
                          frames_per_subseq_num=10,
                          dataset_base_dir=None,
                          trans_thres=0.15,
                          rot_thres=15,
                          frames_range=(0, 0.7),
                          overlap_thres=0.6,
                          interval_skip_frames=1):
    """
    Random select sub set of sequences from scene
    :param scene_frames: scene frames to extract subset
    :param trans_thres_range: translation threshold, based on the center of different frames
    :param max_subseq_num: maximum number of sub sequences
    :param frames_per_subseq_num: for each sub sequences, how many frames in the subset
    :param frames_range: range of start and end within original scene sequences, from (0, 1)
    :param interval_skip_frames: skip interval in original scene frames, used in iteration
    :return: list of selected sub sequences
    """
    assert dataset_base_dir is not None
    n_frames = len(scene_frames)
    if interval_skip_frames < 1:
        interval_skip_frames = 2

    # Simple selection based on trans threshold
    if frames_per_subseq_num * interval_skip_frames > n_frames:
        raise Exception('Not enough frames to be selected')
    rand_start_frame = np.random.randint(int(frames_range[0] * len(scene_frames)),
                                         int(frames_range[1] * len(scene_frames)),
                                         size=max_subseq_num)

    sub_seq_list = []
    dim = scene_frames.get_frame_dim(scene_frames.frames[0])
    K = scene_frames.get_K_mat(scene_frames.frames[0])
    pre_cache_x2d = x_2d_coords(dim[0], dim[1])

    for start_frame_idx in rand_start_frame:
        # print('F:', start_frame_idx)

        # Push start keyframe into frames
        sub_frames = FrameSeqData()
        pre_frame = scene_frames.frames[start_frame_idx]
        sub_frames.frames.append(copy.deepcopy(pre_frame))

        # Iterate the remaining keyframes into subset
        cur_frame_idx = start_frame_idx
        no_found_flag = False
        while cur_frame_idx < n_frames:
            pre_Tcw = sub_frames.get_Tcw(pre_frame)
            pre_depth_path = sub_frames.get_depth_name(pre_frame)
            pre_depth = read_sun3d_depth(os.path.join(dataset_base_dir, pre_depth_path))

            # [Deprecated]
            # pre_img_name = sub_frames.get_image_name(pre_frame)
            # pre_img = cv2.imread(os.path.join(dataset_base_dir, pre_img_name)).astype(np.float32) / 255.0
            # pre_center = camera_center_from_Tcw(pre_Tcw[:3, :3], pre_Tcw[:3, 3])

            pre_search_frame = scene_frames.frames[cur_frame_idx + interval_skip_frames - 1]
            for search_idx in range(cur_frame_idx + interval_skip_frames, n_frames, 1):

                cur_frame = scene_frames.frames[search_idx]
                cur_Tcw = sub_frames.get_Tcw(cur_frame)
                # [Deprecated]
                # cur_center = camera_center_from_Tcw(cur_Tcw[:3, :3], cur_Tcw[:3, 3])
                # cur_img_name = sub_frames.get_image_name(cur_frame)
                # cur_img = cv2.imread(os.path.join(dataset_base_dir, cur_img_name)).astype(np.float32) / 255.0

                rel_angle = rel_rot_angle(pre_Tcw, cur_Tcw)
                rel_dist = rel_distance(pre_Tcw, cur_Tcw)

                overlap = photometric_overlap(pre_depth, K, Ta=pre_Tcw, Tb=cur_Tcw, pre_cache_x2d=pre_cache_x2d)

                # [Deprecated]
                # overlap_map, x_2d = cam_opt.gen_overlap_mask_img(pre_depth, K, Ta=pre_Tcw, Tb=cur_Tcw, pre_cache_x2d=pre_cache_x2d)
                # rel_T = relateive_pose(pre_Tcw[:3, :3], pre_Tcw[:3, 3], cur_Tcw[:3, :3], cur_Tcw[:3, 3])
                # wrap_img, _ = cam_opt.wrapping(pre_img, cur_img, pre_depth, K, rel_T[:3, :3], rel_T[:3, 3])
                # img_list = [
                #     {'img': pre_img},
                #     {'img': cur_img},
                #     {'img': wrap_img},
                #     {'img': overlap_map},
                #     {'img': x_2d[:, :, 0], 'cmap':'gray'},
                #     {'img': x_2d[:, :, 1], 'cmap': 'gray'}
                # ]
                # show_multiple_img(img_list, num_cols=4)
                # plt.show()

                if rel_dist > trans_thres or overlap < overlap_thres or rel_angle > rot_thres:
                    # Select the new keyframe that larger than the trans threshold and add the previous frame as keyframe
                    sub_frames.frames.append(copy.deepcopy(pre_search_frame))
                    pre_frame = pre_search_frame
                    cur_frame_idx = search_idx + 1
                    break
                else:
                    pre_search_frame = cur_frame

                if search_idx == n_frames - 1:
                    no_found_flag = True

            if no_found_flag:
                break

            if len(sub_frames) > frames_per_subseq_num - 1:
                break

        # If the subset is less than setting, ignore
        if len(sub_frames) >= frames_per_subseq_num:
            sub_seq_list.append(sub_frames)

    print('sel: %d', len(sub_seq_list))
    return sub_seq_list


if __name__ == '__main__':
    ori_seq_json_path = '/home/luwei/mnt/Tango/ziqianb/SUN3D/seq.json'
    ori_seq = FrameSeqData(ori_seq_json_path)
    sub_seq_list = rand_sel_subseq_sun3d(ori_seq, trans_thres_range=0.2, rot_thres=15.0, max_subseq_num=20, frames_per_subseq_num=5)

    # out_sub_seq_list = []
    # for seq in sub_seq_list:
    #     frame_instances = copy.deepcopy(seq.frames)
    #     for frame in frame_instances:
    #         frame['extrinsic_Tcw'] = frame['extrinsic_Tcw'].ravel().tolist()
    #         frame['camera_intrinsic'] = frame['camera_intrinsic'].ravel().tolist()
    #     out_sub_seq_list.append(frame_instances)
    # with open('/home/luwei/mnt/Tango/ziqianb/SUN3D/seq_list.json', 'w') as out_json_file:
    #     json.dump(out_sub_seq_list, out_json_file, indent=2)

    # Show 2D seq
    plt.figure()
    ax = plt.gca()
    plot_frames_seq_2d(ori_seq, ax, legend='all')
    for sub_seq in sub_seq_list:
        plot_frames_seq_2d(sub_seq, ax)
    plt.show()