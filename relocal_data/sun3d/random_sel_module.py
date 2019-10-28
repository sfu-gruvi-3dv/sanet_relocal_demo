import os
import sys
import numpy as np
import copy
import cv2
import torch

from scipy.spatial import KDTree

from frame_seq_data import FrameSeqData
import core_3dv.camera_operator as cam_opt
from seq_data.sun3d.read_util import read_sun3d_depth
from evaluator.basic_metric import rel_rot_angle, rel_distance
# from visualizer.visualizer_3d import Visualizer
from libs.pycbf_filter.depth_fill import fill_depth_cross_bf
from relocal_data.sun3d.gen_lmdb_cache import LMDBSeqModel
from seq_data.seven_scenes.read_util import read_7scenese_depth


def sel_triple_sun3d(base_dir, scene_frames, max_triple_num, num_sample_per_triple, trans_thres, overlap_thres):
    """
    Select triples (anchor, positive, negative) from a sun3d sequence
    :param base_dir: dataset base directory
    :param scene_frames: scene frames to extract triples
    :param max_triple_num: maximum number of triples
    :param num_sample_per_triple: number of positive/negative samples per triple
    :param trans_thres: translation threshold for positive samples, based on the center of different frames
    :param overlap_thres: overlap threshold for positive samples, (low, high)
    :return: [{'anchor': frame_dict, 'positive': FrameSeqData, 'negative': FrameSeqData}, {...}, ...]
    """
    dim = scene_frames.get_frame_dim(scene_frames.frames[0])
    K = scene_frames.get_K_mat(scene_frames.frames[0])
    pre_cache_x2d = cam_opt.x_2d_coords(dim[0], dim[1])

    camera_centers = np.empty((len(scene_frames), 3), dtype=np.float32)
    for i, frame in enumerate(scene_frames.frames):
        Tcw = scene_frames.get_Tcw(frame)
        center = cam_opt.camera_center_from_Tcw(Tcw[:3, :3], Tcw[:3, 3])
        camera_centers[i, :] = center

    kdtree = KDTree(camera_centers)

    triple_list = []
    anchor_idces = np.random.choice(len(scene_frames), max_triple_num, replace=False)
    for anchor_idx in anchor_idces:
        anchor_frame = scene_frames.frames[anchor_idx]
        anchor_Tcw = scene_frames.get_Tcw(anchor_frame)
        anchor_depth_path = scene_frames.get_depth_name(anchor_frame)
        anchor_depth = read_sun3d_depth(os.path.join(base_dir, anchor_depth_path))
        anchor_depth[anchor_depth < 1e-5] = 1e-5

        potential_pos_idces = kdtree.query_ball_point(camera_centers[anchor_idx], trans_thres)
        pos_idces = []
        for potential_pos_idx in potential_pos_idces:
            potential_pos_frame = scene_frames.frames[potential_pos_idx]
            potential_pos_Tcw = scene_frames.get_Tcw(potential_pos_frame)
            overlap = cam_opt.photometric_overlap(anchor_depth, K, Ta=anchor_Tcw, Tb=potential_pos_Tcw, pre_cache_x2d=pre_cache_x2d)
            if overlap_thres[0] < overlap < overlap_thres[1]:
                pos_idces.append(potential_pos_idx)

        if len(pos_idces) < num_sample_per_triple:
            continue
        else:
            sel_pos_idces = np.random.choice(pos_idces, num_sample_per_triple, replace=False)

        neg_idces = list(set(range(len(scene_frames))) - set(pos_idces))
        sel_neg_idces = np.random.choice(neg_idces, num_sample_per_triple, replace=False)

        triple_list.append({
            'anchor': copy.deepcopy(anchor_frame),
            'positive': [copy.deepcopy(scene_frames.frames[idx]) for idx in sorted(sel_pos_idces)],
            'negative': [copy.deepcopy(scene_frames.frames[idx]) for idx in sorted(sel_neg_idces)],
        })

        # print(camera_centers[anchor_idx])
        # print(camera_centers[pos_idces])
        # print(camera_centers[neg_idces])
        # print('----------------------------------------------------------')

    return triple_list


def sel_subseq_with_anchor_sun3d(scene_frames,
                                 max_subseq_num,
                                 frames_per_subseq_num=10,
                                 dataset_base_dir=None,
                                 trans_thres=0.15,
                                 rot_thres=15,
                                 frames_range=(0, 0.7),
                                 overlap_thres=0.6,
                                 interval_skip_frames=1,
                                 train_anchor_num=100,
                                 test_anchor_num=100):
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
    max_subseq_num = int(n_frames * max_subseq_num)

    # Simple selection based on trans threshold
    if frames_per_subseq_num * interval_skip_frames > n_frames:
        # raise Exception('Not enough frames to be selected')
        return []
    rand_start_frame = np.random.randint(int(frames_range[0] * len(scene_frames)),
                                         int(frames_range[1] * len(scene_frames)),
                                         size=max_subseq_num)

    sub_seq_list = []
    dim = scene_frames.get_frame_dim(scene_frames.frames[0])
    K = scene_frames.get_K_mat(scene_frames.frames[0])
    pre_cache_x2d = cam_opt.x_2d_coords(dim[0], dim[1])

    for start_frame_idx in rand_start_frame:
        # print('F:', start_frame_idx)

        # Push start keyframe into frames
        sub_frames = FrameSeqData()
        pre_frame = scene_frames.frames[start_frame_idx]
        sub_frames.frames.append(copy.deepcopy(pre_frame))
        sub_frames_idx = [start_frame_idx]

        # Iterate the remaining keyframes into subset
        cur_frame_idx = start_frame_idx
        no_found_flag = False
        while cur_frame_idx + interval_skip_frames < n_frames:
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

                overlap = cam_opt.photometric_overlap(pre_depth, K, Ta=pre_Tcw, Tb=cur_Tcw, pre_cache_x2d=pre_cache_x2d)

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
                # if rel_dist > trans_thres:
                #     print('exceed trans_thres')
                # elif overlap < overlap_thres:
                #     print('exceed overlap_thres')
                # elif rel_angle > rot_thres:
                #     print('exceed rot_thres')

                if rel_dist > trans_thres or overlap < overlap_thres:# or rel_angle > rot_thres:
                    # Select the new keyframe that larger than the trans threshold and add the previous frame as keyframe
                    sub_frames.frames.append(copy.deepcopy(pre_search_frame))
                    pre_frame = pre_search_frame
                    cur_frame_idx = search_idx + 1
                    sub_frames_idx.append(search_idx - 1)
                    break
                else:
                    pre_search_frame = cur_frame

                if search_idx + 1 >= n_frames:
                    no_found_flag = True

            if no_found_flag:
                break

            if len(sub_frames) > frames_per_subseq_num - 1:
                break

        # If the subset is less than setting, ignore
        if len(sub_frames) >= frames_per_subseq_num:
            min_idx = min(sub_frames_idx)
            max_idx = max(sub_frames_idx)
            print(min_idx, max_idx, n_frames)

            train_anchor_frames = []
            train_anchor_idx = []
            for i in range(train_anchor_num):
                anchor_idx = np.random.choice(range(min_idx, max_idx))
                while anchor_idx in sub_frames_idx:
                    anchor_idx = np.random.choice(range(min_idx, max_idx))
                train_anchor_frames.append(scene_frames.frames[anchor_idx])
                train_anchor_idx.append(anchor_idx)

            test_anchor_frames = []
            for i in range(test_anchor_num):
                anchor_idx = np.random.choice(range(min_idx, max_idx))
                while anchor_idx in sub_frames_idx or anchor_idx in train_anchor_idx:
                    anchor_idx = np.random.choice(range(min_idx, max_idx))
                test_anchor_frames.append(scene_frames.frames[anchor_idx])

            sub_seq_list.append({'sub_frames': sub_frames, 'train_anchor_frames': train_anchor_frames, 'test_anchor_frames': test_anchor_frames})

    print('sel: %d', len(sub_seq_list))
    return sub_seq_list


def sel_pairs_with_overlap_range_sun3d(scene_frames,
                                       scene_lmdb: LMDBSeqModel,
                                       max_subseq_num,
                                       frames_per_subseq_num=10,
                                       dataset_base_dir=None,
                                       trans_thres=0.15,
                                       rot_thres=15,
                                       frames_range=(0, 0.7),
                                       overlap_thres=0.5,
                                       scene_dist_thres=(0.0, 1.0),
                                       interval_skip_frames=1,
                                       train_anchor_num=100,
                                       test_anchor_num=100):
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
    use_lmdb_cache = True if scene_lmdb is not None else False

    assert dataset_base_dir is not None
    n_frames = len(scene_frames)
    if interval_skip_frames < 1:
        interval_skip_frames = 2
    max_subseq_num = int(n_frames * max_subseq_num)

    # Simple selection based on trans threshold
    # if frames_per_subseq_num * interval_skip_frames > n_frames:
    #     # raise Exception('Not enough frames to be selected')
    #     return []
    rand_start_frame = np.random.randint(int(frames_range[0] * len(scene_frames)),
                                         int(frames_range[1] * len(scene_frames)),
                                         size=max_subseq_num)

    sub_seq_list = []
    dim = scene_frames.get_frame_dim(scene_frames.frames[0])
    dim = list(dim)
    dim[0] = int(dim[0] // 4)
    dim[1] = int(dim[1] // 4)
    K = scene_frames.get_K_mat(scene_frames.frames[0])
    K /= 4.0
    K[2, 2] = 1.0
    pre_cache_x2d = cam_opt.x_2d_coords(dim[0], dim[1])

    for start_frame_idx in rand_start_frame:
        # print('F:', start_frame_idx)

        # Push start keyframe into frames
        sub_frames = FrameSeqData()
        pre_frame = scene_frames.frames[start_frame_idx]
        sub_frames.frames.append(copy.deepcopy(pre_frame))
        sub_frames_idx = [start_frame_idx]

        # Iterate the remaining keyframes into subset
        cur_frame_idx = start_frame_idx
        no_found_flag = False
        while cur_frame_idx + interval_skip_frames < n_frames:
            pre_Tcw = sub_frames.get_Tcw(pre_frame)
            pre_depth_path = sub_frames.get_depth_name(pre_frame)
            # pre_depth = read_sun3d_depth(os.path.join(dataset_base_dir, pre_depth_path))
            pre_depth = scene_lmdb.read_depth(pre_depth_path) if use_lmdb_cache else \
                read_sun3d_depth(os.path.join(dataset_base_dir, pre_depth_path))
            pre_depth = cv2.resize(pre_depth, (dim[1], dim[0]), interpolation=cv2.INTER_NEAREST)
            # H, W = pre_depth.shape
            # if float(np.sum(pre_depth <= 1e-5)) / float(H*W) > 0.2:
            #     continue
            # pre_depth = torch.from_numpy(pre_depth).cuda()
            # pre_Tcw_gpu = torch.from_numpy(pre_Tcw).cuda()
            # pre_img_name = sub_frames.get_image_name(pre_frame)
            # pre_img = cv2.imread(os.path.join(dataset_base_dir, pre_img_name))
            # pre_depth = fill_depth_cross_bf(pre_img, pre_depth)

            # [Deprecated]
            # import cv2
            # pre_img_name = sub_frames.get_image_name(pre_frame)
            # pre_img = cv2.imread(os.path.join(dataset_base_dir, pre_img_name)).astype(np.float32) / 255.0
            # pre_center = cam_opt.camera_center_from_Tcw(pre_Tcw[:3, :3], pre_Tcw[:3, 3])

            pre_search_frame = scene_frames.frames[cur_frame_idx + interval_skip_frames - 1]
            for search_idx in range(cur_frame_idx + interval_skip_frames, n_frames, 1):

                cur_frame = scene_frames.frames[search_idx]
                cur_Tcw = sub_frames.get_Tcw(cur_frame)
                # cur_Tcw_gpu = torch.from_numpy(cur_Tcw).cuda()
                # cur_depth_path = sub_frames.get_depth_name(cur_frame)
                # cur_depth = read_sun3d_depth(os.path.join(dataset_base_dir, cur_depth_path))
                # H, W = cur_depth.shape

                # [Deprecated]
                # cur_center = cam_opt.camera_center_from_Tcw(cur_Tcw[:3, :3], cur_Tcw[:3, 3])
                # cur_img_name = sub_frames.get_image_name(cur_frame)
                # cur_img = cv2.imread(os.path.join(dataset_base_dir, cur_img_name)).astype(np.float32) / 255.0

                rel_angle = rel_rot_angle(pre_Tcw, cur_Tcw)
                rel_dist = rel_distance(pre_Tcw, cur_Tcw)

                overlap = cam_opt.photometric_overlap(pre_depth, K, Ta=pre_Tcw, Tb=cur_Tcw, pre_cache_x2d=pre_cache_x2d)

                # mean scene coordinate dist
                # pre_Twc = cam_opt.camera_pose_inv(R=pre_Tcw[:3, :3], t=pre_Tcw[:3, 3])
                # d_a = pre_depth.reshape((H * W, 1))
                # x_a_2d = pre_cache_x2d.reshape((H * W, 2))
                # X_3d = cam_opt.pi_inv(K, x_a_2d, d_a)
                # pre_X_3d = cam_opt.transpose(pre_Twc[:3, :3], pre_Twc[:3, 3], X_3d).reshape((H, W, 3))
                # pre_mean = np.empty((3,), dtype=np.float)
                # pre_mean[0] = np.mean(pre_X_3d[pre_depth > 1e-5, 0])
                # pre_mean[1] = np.mean(pre_X_3d[pre_depth > 1e-5, 1])
                # pre_mean[2] = np.mean(pre_X_3d[pre_depth > 1e-5, 2])
                #
                # cur_Twc = cam_opt.camera_pose_inv(R=cur_Tcw[:3, :3], t=cur_Tcw[:3, 3])
                # d_a = cur_depth.reshape((H * W, 1))
                # x_a_2d = pre_cache_x2d.reshape((H * W, 2))
                # X_3d = cam_opt.pi_inv(K, x_a_2d, d_a)
                # cur_X_3d = cam_opt.transpose(cur_Twc[:3, :3], cur_Twc[:3, 3], X_3d).reshape((H, W, 3))
                # cur_mean = np.empty((3,), dtype=np.float)
                # cur_mean[0] = np.mean(cur_X_3d[cur_depth > 1e-5, 0])
                # cur_mean[1] = np.mean(cur_X_3d[cur_depth > 1e-5, 1])
                # cur_mean[2] = np.mean(cur_X_3d[cur_depth > 1e-5, 2])
                #
                # scene_dist = np.linalg.norm(pre_mean - cur_mean)

                # def keyPressEvent(obj, event):
                #     key = obj.GetKeySym()
                #     if key == 'Left':
                #         tmp_img = pre_img
                #         X_3d = pre_X_3d.reshape((H * W, 3))
                #         vis.set_point_cloud(X_3d, tmp_img.reshape((H * W, 3)))
                #         # vis.add_frame_pose(cur_Tcw[:3, :3], cur_Tcw[:3, 3])
                #
                #     if key == 'Right':
                #         tmp_img = cur_img
                #         X_3d = cur_X_3d.reshape((H * W, 3))
                #         vis.set_point_cloud(X_3d, tmp_img.reshape((H * W, 3)))
                #         # vis.add_frame_pose(cur_Tcw[:3, :3], cur_Tcw[:3, 3])
                #
                #     if key == 'Up':
                #         vis.set_point_cloud(pre_mean.reshape((1, 3)), pt_size=10)
                #
                #     if key == 'Down':
                #         vis.set_point_cloud(cur_mean.reshape((1, 3)), pt_size=10)
                #     return
                # vis = Visualizer(1280, 720)
                # vis.bind_keyboard_event(keyPressEvent)
                # vis.show()
                # vis.close()

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
                # if rel_dist > trans_thres:
                #     print('exceed trans_thres')
                # elif overlap < overlap_thres:
                #     print('exceed overlap_thres')
                # elif rel_angle > rot_thres:
                #     print('exceed rot_thres')

                # if overlap_thres[0] <= overlap <= overlap_thres[1] and \
                #    rot_thres[0] <= rel_angle <= rot_thres[1]: #and \
                #     # scene_dist_thres[0] <= scene_dist <= scene_dist_thres[1]:
                #     sub_frames.frames.append(copy.deepcopy(cur_frame))

                if overlap < overlap_thres or rel_dist > trans_thres: #or scene_dist > scene_dist_thres[1]:
                    # Select the new keyframe that larger than the trans threshold and add the previous frame as keyframe
                    sub_frames.frames.append(copy.deepcopy(pre_search_frame))
                    pre_frame = pre_search_frame
                    cur_frame_idx = search_idx + 1
                    sub_frames_idx.append(search_idx - 1)
                    break
                else:
                    pre_search_frame = cur_frame

                if search_idx + 1 >= n_frames:
                    no_found_flag = True

            if no_found_flag:
                break

            if len(sub_frames) > frames_per_subseq_num - 1:
                break

        # If the subset is less than setting, ignore
        if len(sub_frames) >= frames_per_subseq_num:
            min_idx = min(sub_frames_idx)
            max_idx = max(sub_frames_idx)
            print(min_idx, max_idx, n_frames)
            # factor = (max_idx - min_idx) // 3
            #
            # min_Tcw = sub_frames.get_Tcw(sub_frames.frames[0])
            # max_Tcw = sub_frames.get_Tcw(sub_frames.frames[-1])
            potential_anchor_idces = []
            # for i in range(min_idx + factor, max_idx - factor, 1):
            #     cur_frame = scene_frames.frames[i]
            #     cur_Tcw = scene_frames.get_Tcw(cur_frame)
            #     cur_depth_path = sub_frames.get_depth_name(cur_frame)
            #     cur_depth = scene_lmdb.read_depth(cur_depth_path)
            #     cur_depth = cv2.resize(cur_depth, (dim[1], dim[0]), interpolation=cv2.INTER_NEAREST)
            #     H, W = cur_depth.shape
            #     if float(np.sum(cur_depth <= 1e-5)) / float(H*W) > 0.2:
            #         continue
            #     min_overlap = cam_opt.photometric_overlap(cur_depth, K, Ta=cur_Tcw, Tb=min_Tcw,
            #                                               pre_cache_x2d=pre_cache_x2d)
            #     max_overlap = cam_opt.photometric_overlap(cur_depth, K, Ta=cur_Tcw, Tb=max_Tcw,
            #                                               pre_cache_x2d=pre_cache_x2d)
            #     min_rel_angle = rel_rot_angle(cur_Tcw, min_Tcw)
            #     max_rel_angle = rel_rot_angle(cur_Tcw, max_Tcw)
            #     if min_overlap < 0.65 and max_overlap < 0.65 and \
            #        ((0.5 < min_overlap and min_rel_angle < 20.0) or \
            #        (0.5 < max_overlap and max_rel_angle < 20.0)):
            #         potential_anchor_idces.append(i)
            for i in range(min_idx, max_idx):
                if i not in sub_frames_idx:
                    potential_anchor_idces.append(i)

            if len(potential_anchor_idces) >= train_anchor_num + test_anchor_num:
                anchor_idces = np.random.choice(range(len(potential_anchor_idces)),
                                                size=train_anchor_num + test_anchor_num, replace=False)

                train_anchor_frames = []
                for i in anchor_idces[:train_anchor_num]:
                    train_anchor_frames.append(scene_frames.frames[potential_anchor_idces[i]])

                test_anchor_frames = []
                for i in anchor_idces[train_anchor_num:]:
                    test_anchor_frames.append(scene_frames.frames[potential_anchor_idces[i]])

                sub_seq_list.append({'sub_frames': sub_frames, 'train_anchor_frames': train_anchor_frames,
                                     'test_anchor_frames': test_anchor_frames})
                print('selected', len(potential_anchor_idces), len(sub_frames))

    print('sel: %d', len(sub_seq_list))
    return sub_seq_list


def sel_pairs_with_overlap_range_7scene(scene_frames,
                                       scene_lmdb: LMDBSeqModel,
                                       max_subseq_num,
                                       frames_per_subseq_num=10,
                                       dataset_base_dir=None,
                                       trans_thres=0.15,
                                       rot_thres=15,
                                       frames_range=(0, 0.7),
                                       overlap_thres=0.5,
                                       scene_dist_thres=(0.0, 1.0),
                                       interval_skip_frames=1,
                                       train_anchor_num=100,
                                       test_anchor_num=100):
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
    use_lmdb_cache = True if scene_lmdb is not None else False

    assert dataset_base_dir is not None
    n_frames = len(scene_frames)
    if interval_skip_frames < 1:
        interval_skip_frames = 2
    max_subseq_num = int(n_frames * max_subseq_num)

    # Simple selection based on trans threshold
    # if frames_per_subseq_num * interval_skip_frames > n_frames:
    #     # raise Exception('Not enough frames to be selected')
    #     return []
    rand_start_frame = np.random.randint(int(frames_range[0] * len(scene_frames)),
                                         int(frames_range[1] * len(scene_frames)),
                                         size=max_subseq_num)

    sub_seq_list = []
    dim = scene_frames.get_frame_dim(scene_frames.frames[0])
    dim = list(dim)
    dim[0] = int(dim[0] // 4)
    dim[1] = int(dim[1] // 4)
    K = scene_frames.get_K_mat(scene_frames.frames[0])
    K /= 4.0
    K[2, 2] = 1.0
    pre_cache_x2d = cam_opt.x_2d_coords(dim[0], dim[1])

    for start_frame_idx in rand_start_frame:
        # print('F:', start_frame_idx)

        # Push start keyframe into frames
        sub_frames = FrameSeqData()
        pre_frame = scene_frames.frames[start_frame_idx]
        sub_frames.frames.append(copy.deepcopy(pre_frame))
        sub_frames_idx = [start_frame_idx]

        # Iterate the remaining keyframes into subset
        cur_frame_idx = start_frame_idx
        no_found_flag = False
        while cur_frame_idx + interval_skip_frames < n_frames:
            pre_Tcw = sub_frames.get_Tcw(pre_frame)
            pre_depth_path = sub_frames.get_depth_name(pre_frame)
            # pre_depth = read_sun3d_depth(os.path.join(dataset_base_dir, pre_depth_path))
            pre_depth = scene_lmdb.read_depth(pre_depth_path) if use_lmdb_cache else \
                read_7scenese_depth(os.path.join(dataset_base_dir, pre_depth_path))
            pre_depth = cv2.resize(pre_depth, (dim[1], dim[0]), interpolation=cv2.INTER_NEAREST)
            # H, W = pre_depth.shape
            # if float(np.sum(pre_depth <= 1e-5)) / float(H*W) > 0.2:
            #     continue
            # pre_depth = torch.from_numpy(pre_depth).cuda()
            # pre_Tcw_gpu = torch.from_numpy(pre_Tcw).cuda()
            # pre_img_name = sub_frames.get_image_name(pre_frame)
            # pre_img = cv2.imread(os.path.join(dataset_base_dir, pre_img_name))
            # pre_depth = fill_depth_cross_bf(pre_img, pre_depth)

            # [Deprecated]
            # import cv2
            # pre_img_name = sub_frames.get_image_name(pre_frame)
            # pre_img = cv2.imread(os.path.join(dataset_base_dir, pre_img_name)).astype(np.float32) / 255.0
            # pre_center = cam_opt.camera_center_from_Tcw(pre_Tcw[:3, :3], pre_Tcw[:3, 3])

            pre_search_frame = scene_frames.frames[cur_frame_idx + interval_skip_frames - 1]
            for search_idx in range(cur_frame_idx + interval_skip_frames, n_frames, 1):

                cur_frame = scene_frames.frames[search_idx]
                cur_Tcw = sub_frames.get_Tcw(cur_frame)
                # cur_Tcw_gpu = torch.from_numpy(cur_Tcw).cuda()
                # cur_depth_path = sub_frames.get_depth_name(cur_frame)
                # cur_depth = read_sun3d_depth(os.path.join(dataset_base_dir, cur_depth_path))
                # H, W = cur_depth.shape

                # [Deprecated]
                # cur_center = cam_opt.camera_center_from_Tcw(cur_Tcw[:3, :3], cur_Tcw[:3, 3])
                # cur_img_name = sub_frames.get_image_name(cur_frame)
                # cur_img = cv2.imread(os.path.join(dataset_base_dir, cur_img_name)).astype(np.float32) / 255.0

                rel_angle = rel_rot_angle(pre_Tcw, cur_Tcw)
                rel_dist = rel_distance(pre_Tcw, cur_Tcw)

                overlap = cam_opt.photometric_overlap(pre_depth, K, Ta=pre_Tcw, Tb=cur_Tcw, pre_cache_x2d=pre_cache_x2d)

                # mean scene coordinate dist
                # pre_Twc = cam_opt.camera_pose_inv(R=pre_Tcw[:3, :3], t=pre_Tcw[:3, 3])
                # d_a = pre_depth.reshape((H * W, 1))
                # x_a_2d = pre_cache_x2d.reshape((H * W, 2))
                # X_3d = cam_opt.pi_inv(K, x_a_2d, d_a)
                # pre_X_3d = cam_opt.transpose(pre_Twc[:3, :3], pre_Twc[:3, 3], X_3d).reshape((H, W, 3))
                # pre_mean = np.empty((3,), dtype=np.float)
                # pre_mean[0] = np.mean(pre_X_3d[pre_depth > 1e-5, 0])
                # pre_mean[1] = np.mean(pre_X_3d[pre_depth > 1e-5, 1])
                # pre_mean[2] = np.mean(pre_X_3d[pre_depth > 1e-5, 2])
                #
                # cur_Twc = cam_opt.camera_pose_inv(R=cur_Tcw[:3, :3], t=cur_Tcw[:3, 3])
                # d_a = cur_depth.reshape((H * W, 1))
                # x_a_2d = pre_cache_x2d.reshape((H * W, 2))
                # X_3d = cam_opt.pi_inv(K, x_a_2d, d_a)
                # cur_X_3d = cam_opt.transpose(cur_Twc[:3, :3], cur_Twc[:3, 3], X_3d).reshape((H, W, 3))
                # cur_mean = np.empty((3,), dtype=np.float)
                # cur_mean[0] = np.mean(cur_X_3d[cur_depth > 1e-5, 0])
                # cur_mean[1] = np.mean(cur_X_3d[cur_depth > 1e-5, 1])
                # cur_mean[2] = np.mean(cur_X_3d[cur_depth > 1e-5, 2])
                #
                # scene_dist = np.linalg.norm(pre_mean - cur_mean)

                # def keyPressEvent(obj, event):
                #     key = obj.GetKeySym()
                #     if key == 'Left':
                #         tmp_img = pre_img
                #         X_3d = pre_X_3d.reshape((H * W, 3))
                #         vis.set_point_cloud(X_3d, tmp_img.reshape((H * W, 3)))
                #         # vis.add_frame_pose(cur_Tcw[:3, :3], cur_Tcw[:3, 3])
                #
                #     if key == 'Right':
                #         tmp_img = cur_img
                #         X_3d = cur_X_3d.reshape((H * W, 3))
                #         vis.set_point_cloud(X_3d, tmp_img.reshape((H * W, 3)))
                #         # vis.add_frame_pose(cur_Tcw[:3, :3], cur_Tcw[:3, 3])
                #
                #     if key == 'Up':
                #         vis.set_point_cloud(pre_mean.reshape((1, 3)), pt_size=10)
                #
                #     if key == 'Down':
                #         vis.set_point_cloud(cur_mean.reshape((1, 3)), pt_size=10)
                #     return
                # vis = Visualizer(1280, 720)
                # vis.bind_keyboard_event(keyPressEvent)
                # vis.show()
                # vis.close()

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
                # if rel_dist > trans_thres:
                #     print('exceed trans_thres')
                # elif overlap < overlap_thres:
                #     print('exceed overlap_thres')
                # elif rel_angle > rot_thres:
                #     print('exceed rot_thres')

                # if overlap_thres[0] <= overlap <= overlap_thres[1] and \
                #    rot_thres[0] <= rel_angle <= rot_thres[1]: #and \
                #     # scene_dist_thres[0] <= scene_dist <= scene_dist_thres[1]:
                #     sub_frames.frames.append(copy.deepcopy(cur_frame))

                if overlap < overlap_thres or rel_dist > trans_thres: #or scene_dist > scene_dist_thres[1]:
                    # Select the new keyframe that larger than the trans threshold and add the previous frame as keyframe
                    sub_frames.frames.append(copy.deepcopy(pre_search_frame))
                    pre_frame = pre_search_frame
                    cur_frame_idx = search_idx + 1
                    sub_frames_idx.append(search_idx - 1)
                    break
                else:
                    pre_search_frame = cur_frame

                if search_idx + 1 >= n_frames:
                    no_found_flag = True

            if no_found_flag:
                break

            if len(sub_frames) > frames_per_subseq_num - 1:
                break

        # If the subset is less than setting, ignore
        if len(sub_frames) >= frames_per_subseq_num:
            min_idx = min(sub_frames_idx)
            max_idx = max(sub_frames_idx)
            print(min_idx, max_idx, n_frames)
            # factor = (max_idx - min_idx) // 3
            #
            # min_Tcw = sub_frames.get_Tcw(sub_frames.frames[0])
            # max_Tcw = sub_frames.get_Tcw(sub_frames.frames[-1])
            potential_anchor_idces = []
            # for i in range(min_idx + factor, max_idx - factor, 1):
            #     cur_frame = scene_frames.frames[i]
            #     cur_Tcw = scene_frames.get_Tcw(cur_frame)
            #     cur_depth_path = sub_frames.get_depth_name(cur_frame)
            #     cur_depth = scene_lmdb.read_depth(cur_depth_path)
            #     cur_depth = cv2.resize(cur_depth, (dim[1], dim[0]), interpolation=cv2.INTER_NEAREST)
            #     H, W = cur_depth.shape
            #     if float(np.sum(cur_depth <= 1e-5)) / float(H*W) > 0.2:
            #         continue
            #     min_overlap = cam_opt.photometric_overlap(cur_depth, K, Ta=cur_Tcw, Tb=min_Tcw,
            #                                               pre_cache_x2d=pre_cache_x2d)
            #     max_overlap = cam_opt.photometric_overlap(cur_depth, K, Ta=cur_Tcw, Tb=max_Tcw,
            #                                               pre_cache_x2d=pre_cache_x2d)
            #     min_rel_angle = rel_rot_angle(cur_Tcw, min_Tcw)
            #     max_rel_angle = rel_rot_angle(cur_Tcw, max_Tcw)
            #     if min_overlap < 0.65 and max_overlap < 0.65 and \
            #        ((0.5 < min_overlap and min_rel_angle < 20.0) or \
            #        (0.5 < max_overlap and max_rel_angle < 20.0)):
            #         potential_anchor_idces.append(i)
            for i in range(min_idx, max_idx):
                if i not in sub_frames_idx:
                    potential_anchor_idces.append(i)

            if len(potential_anchor_idces) >= train_anchor_num + test_anchor_num:
                anchor_idces = np.random.choice(range(len(potential_anchor_idces)),
                                                size=train_anchor_num + test_anchor_num, replace=False)

                train_anchor_frames = []
                for i in anchor_idces[:train_anchor_num]:
                    train_anchor_frames.append(scene_frames.frames[potential_anchor_idces[i]])

                test_anchor_frames = []
                for i in anchor_idces[train_anchor_num:]:
                    test_anchor_frames.append(scene_frames.frames[potential_anchor_idces[i]])

                sub_seq_list.append({'sub_frames': sub_frames, 'train_anchor_frames': train_anchor_frames,
                                     'test_anchor_frames': test_anchor_frames})
                print('selected', len(potential_anchor_idces), len(sub_frames))

    print('sel: %d', len(sub_seq_list))
    return sub_seq_list


if __name__ == '__main__':
    base_dir = '/mnt/Exp_2/SUN3D_Valid'
    seq_name = 'home_han/apartment_han_oct_31_2012_scan1_erika'
    scene_frames = FrameSeqData(os.path.join(base_dir, seq_name, 'seq.json'))
    triple_list = sel_triple_sun3d(base_dir, scene_frames, max_triple_num=1, num_sample_per_triple=10, trans_thres=0.1, overlap_thres=0.8)
