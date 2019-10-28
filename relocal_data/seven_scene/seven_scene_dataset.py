import numpy as np
import pickle
import cv2
import random
import os
import torch

from collections import namedtuple
from visualizer.visualizer_2d import show_multiple_img
from torch.utils.data import Dataset, DataLoader
from libs.pycbf_filter.depth_fill import fill_depth_cross_bf
from img_proc.img_dim import crop_by_intrinsic
from core_3dv.camera_operator import *
from frame_seq_data import K_from_frame, FrameSeqData
from core_math.transfom import quaternion_from_matrix
from relocal_data.sun3d.random_sel_module import sel_pairs_with_overlap_range_7scene
from seq_data.seven_scenes.read_util import *

SevenSceneSeqFilterParam = namedtuple('SevenSceneSeqFilterParam', field_names=[
    'trans_thres',
    'rotation_threshold',
    'overlap_threshold',
    'scene_dist_threshold',
    'max_sub_seq_per_scene',
    'frames_per_sub_seq',
    'skip_frames',
    'shuffle_list',
    'train_anchor_num',
    'test_anchor_num'])

SevenSceneSeqFilterParam.__new__.__defaults__ = (2.0, (20.0, 999.0), 0.2, (0.0, 1.0), 0.05, 5, 1, False, 200, 0)

class SevenSceneDataset(Dataset):

    @staticmethod
    def cache_file_name(seq_lists, sel_params: SevenSceneSeqFilterParam):
        """
        Generate cache file name based on sequence lists and the filtering parameters, used for locating or creating cache file.
        :param seq_lists: sequences lists
        :param sel_params: filtering parameters
        :return: cache file name
        """
        seq_lists = sorted(seq_lists)
        seq_name_string = ""
        for seq_name in seq_lists:
            seq_name_string += seq_name[:2]
            seq_name_string += seq_name[-2:]

        param_string = ""
        for field_name in sel_params._fields:
            param_attr = getattr(sel_params, field_name)
            if isinstance(param_attr, tuple) or isinstance(param_attr, list):
                param_attr = [str(v).replace(" ", "_") for v in param_attr]
                param_string += "%s_" % field_name[:10]
                for param_attr_item in param_attr:
                    param_string += (param_attr_item[:5] + "_")
            else:
                param_attr = str(param_attr).replace(" ", "_")
                param_string += "%s_%s" % (field_name[:10], param_attr[:5])

        file_name = seq_name_string + '_' + param_string
        return file_name if len(file_name) < 255 else file_name[:255]

    @staticmethod
    def gen_frame_list(seq_dir, seq_lists, sel_params):
        total_sub_seq_list = []

        for seq_name in seq_lists:
            in_frame_path = os.path.join(seq_dir, seq_name, 'seq.json')

            if os.path.exists(in_frame_path):
                frames = FrameSeqData(in_frame_path)
                sub_frames_list = sel_pairs_with_overlap_range_7scene(frames, scene_lmdb=None,
                                                                     trans_thres=sel_params.trans_thres,
                                                                     rot_thres=sel_params.rotation_threshold,
                                                                     dataset_base_dir=seq_dir,
                                                                     overlap_thres=sel_params.overlap_threshold,
                                                                     scene_dist_thres=sel_params.scene_dist_threshold,
                                                                     max_subseq_num=sel_params.max_sub_seq_per_scene,
                                                                     frames_per_subseq_num=sel_params.frames_per_sub_seq,
                                                                     frames_range=(0.02, 0.8),
                                                                     interval_skip_frames=sel_params.skip_frames,
                                                                     train_anchor_num=sel_params.train_anchor_num,
                                                                     test_anchor_num=sel_params.test_anchor_num)
                total_sub_seq_list += sub_frames_list

        return total_sub_seq_list

    def __init__(self, base_dir, seq_name_lists, workspace_dir, sel_params:SevenSceneSeqFilterParam, use_offline_cache=True,
                 transform=None, rand_flip=False, fill_depth_holes=True, output_dim=(3, 240, 320), supervised_tag='train_anchor_frames'):
        super(SevenSceneDataset, self).__init__()

        self.base_dir = base_dir
        self.random_flip = rand_flip
        self.transform_func = transform
        self.output_dim = output_dim
        self.supervised_out_tag = supervised_tag
        self.fill_depth_holes = fill_depth_holes

        self.workspace_dir = workspace_dir
        self.seq_name_lists = seq_name_lists
        self.sel_param = sel_params

        self.depth_k = np.asarray([[585, 0, 320], [0, 585, 240], [0, 0, 1]], dtype=np.float32)
        self.img_k = np.asarray([[525, 0, 324], [0, 525, 244], [0, 0, 1]], dtype=np.float32)

        if not os.path.exists(workspace_dir):
            os.mkdir(workspace_dir)

        if use_offline_cache:
            file_path = os.path.join(workspace_dir, self.cache_file_name(seq_name_lists, sel_params))

            # load the offline cache if the file is exist
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    self.seq_list = pickle.load(f, encoding='latin1')
            else:
                self.seq_list = self.gen_frame_list(base_dir, seq_name_lists, sel_params)
                with open(file_path, 'wb') as out_f:
                    pickle.dump(self.seq_list, out_f)
                    # print('Found sequences frames, total:', len(file_path))
                    print('Saved to %s' % file_path)
        else:
            self.seq_list = self.gen_frame_list(base_dir, seq_name_lists, sel_params)

    def __len__(self):
        return len(self.seq_list)

    def load_frame_2_tensors(self, frame, out_frame_dim):
        C, H, W = out_frame_dim
        K = self.depth_k.copy()

        Tcw = np.asarray(frame['extrinsic_Tcw'], dtype=np.float32).reshape((3, 4))
        Rcw, tcw = Tcw[:3, :3], Tcw[:3, 3]
        img_file_name = frame['file_name']
        depth_file_name = frame['depth_file_name']

        # Load image and depth
        img = cv2.imread(os.path.join(self.base_dir, img_file_name))
        ori_H, ori_W, _ = img.shape
        img = crop_by_intrinsic(img, self.img_k, self.depth_k)
        img = cv2.resize(img, (ori_W, ori_H))
        depth = read_7scenese_depth(os.path.join(self.base_dir, depth_file_name))

        # Post-process image and depth (fill the holes with cross bilateral filter)
        resize_ratio = max(H / ori_H, W / ori_W)
        img = cv2.resize(img, dsize=(int(resize_ratio * ori_W), int(resize_ratio * ori_H)))
        depth = cv2.resize(depth, dsize=(int(resize_ratio * ori_W), int(resize_ratio * ori_H)), interpolation=cv2.INTER_NEAREST)
        if self.fill_depth_holes:
            depth = fill_depth_cross_bf(img, depth)
        depth[depth < 1e-5] = 1e-5
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # camera intrinsic parameters:
        K[0, 0] *= resize_ratio
        K[0, 2] = (resize_ratio * ori_W) / 2
        K[1, 1] *= resize_ratio
        K[1, 2] = (resize_ratio * ori_H) / 2
        new_K = K.copy()
        new_K[0, 2] = W / 2
        new_K[1, 2] = H / 2

        # crop and resize with new K
        img = crop_by_intrinsic(img, K, new_K)
        depth = crop_by_intrinsic(depth, K, new_K, interp_method='nearest')

        # camera motion representation: (center, rotation_center2world)
        c = camera_center_from_Tcw(Rcw, tcw)
        Rwc = np.eye(4)
        Rwc[:3, :3] = Rcw.T
        q = quaternion_from_matrix(Rwc)
        log_q = log_quat(q)
        pose_vector = np.concatenate((c, log_q)).astype(np.float32)

        # convert to torch.tensor
        ori_img_tensor = torch.from_numpy(img.transpose((2, 0, 1)))  # (C, H, W)
        img_tensor = ori_img_tensor.clone()
        if self.transform_func:
            img_tensor = self.transform_func(img_tensor)
        depth_tensor = torch.from_numpy(depth).view(1, H, W)  # (1, H, W)

        pose_vector = torch.from_numpy(pose_vector)     # (1, 3)
        Tcw_tensor = torch.from_numpy(Tcw)    # (3, 4)
        K_tensor = torch.from_numpy(new_K)  # (3, 3)

        return pose_vector, img_tensor, depth_tensor, K_tensor, Tcw_tensor, ori_img_tensor

    def __getitem__(self, idx):
        input_frames = self.seq_list[idx]['sub_frames'].frames
        C, H, W = self.output_dim

        rand_flip_flag = np.random.randint(2) if self.random_flip else 0
        if rand_flip_flag == 1:
            input_frames = input_frames[::-1]

        # Load frames
        pose_vectors = []
        img_tensors = []
        depth_tensors = []
        K_tensors = []
        Tcw_tensors = []
        ori_img_tensors = []
        for frame in input_frames:
            pose_vector, img_tensor, depth_tensor, K_tensor, Tcw_tensor, ori_img_tensor = self.load_frame_2_tensors(frame, self.output_dim)
            pose_vectors.append(pose_vector)
            img_tensors.append(img_tensor)
            depth_tensors.append(depth_tensor)
            K_tensors.append(K_tensor)
            Tcw_tensors.append(Tcw_tensor)
            ori_img_tensors.append(ori_img_tensor)
        pose_vectors = torch.stack(pose_vectors, dim=0)
        img_tensors = torch.stack(img_tensors, dim=0)
        depth_tensors = torch.stack(depth_tensors, dim=0)
        K_tensors = torch.stack(K_tensors, dim=0)
        Tcw_tensors = torch.stack(Tcw_tensors, dim=0)
        ori_img_tensors = torch.stack(ori_img_tensors, dim=0)

        # Sample a item
        sample = random.choice(self.seq_list[idx][self.supervised_out_tag])
        pose_vector, img_tensor, depth_tensor, K_tensor, Tcw_tensor, ori_img_tensor = self.load_frame_2_tensors(sample, self.output_dim)

        return {'frames_img': img_tensors, 'frames_depth': depth_tensors, 'frames_pose': pose_vectors, 'frames_K': K_tensors, 'frames_Tcw': Tcw_tensors, 'frames_ori_img': ori_img_tensors,
                'img': img_tensor, 'depth': depth_tensor, 'pose': pose_vector, 'K': K_tensor, 'Tcw': Tcw_tensor, 'ori_img': ori_img_tensor}


if __name__ == '__main__':

    param = SevenSceneSeqFilterParam()
    data_set = SevenSceneDataset(base_dir='/home/ziqianb/Desktop/datasets/7scenes/',
                                 seq_name_lists=['chess/seq-04', 'redkitchen/seq-07'],
                                 workspace_dir='/home/ziqianb/Desktop/datasets/7scenes/tmp',
                                 sel_params=param, use_offline_cache=True)

    data_loader = DataLoader(data_set, batch_size=1, num_workers=0, shuffle=True)
    print('size of the dataset:', len(data_set))

    for seq_dict in data_loader:
        Tcw = seq_dict['frames_Tcw'][0].numpy()
        I = seq_dict['frames_img'][0].numpy().transpose(0, 2, 3, 1)
        depth = seq_dict['frames_depth'][0, :, 0].numpy()
        K = seq_dict['frames_K'][0].numpy()

        query_img = seq_dict['img'][0].numpy().transpose(1, 2, 0)  # depth[0 + 1].reshape((256, 256, 1))
        query_depth = seq_dict['depth'][0, 0].numpy()
        query_Tcw = seq_dict['Tcw'][0].numpy()

        img_list = []
        wrap_list = []
        depth_list = []
        for i in range(min(I.shape[0] - 1, 5)):
            cur_img = I[i]
            next_img = I[i+1]
            cur_K = K[i]
            rel_T = relateive_pose(Tcw[i, :3, :3], Tcw[i, :3, 3], Tcw[i+1, :3, :3], Tcw[i+1, :3, 3])
            next2cur, _ = wrapping(cur_img, next_img, depth[i], cur_K, rel_T[:3, :3], rel_T[:3, 3])
            img_list.append({'img': cur_img, 'title': str(i)})
            depth_list.append({'img': depth[i], 'title': str(i)})
            wrap_list.append({'img': next2cur, 'title': str(i+1) + ' to ' + str(i)})
            if i == min(I.shape[0] - 1, 5) - 1:
                img_list.append({'img': next_img, 'title': str(i+1)})
                depth_list.append({'img': depth[i+1], 'title': str(i+1)})

        wrap_list.append({'img': query_img, 'title': 'query image'})
        show_multiple_img(img_list + wrap_list + depth_list, title='dataset debug', num_cols=max(2, min(I.shape[0], 5)))

        # cur_img = I[0]#.reshape((256, 256, 1))
        # next_img = seq_dict['img'][0].numpy().transpose(1, 2, 0)#depth[0 + 1].reshape((256, 256, 1))
        # cur_depth = depth[0]
        # cur_K = K[0]
        # Tcw_next = seq_dict['Tcw'][0].numpy()
        # rel_T = relateive_pose(Tcw[0, :3, :3], Tcw[0, :3, 3], Tcw_next[:3, :3], Tcw_next[:3, 3])
        # next2cur, _ = wrapping(cur_img, next_img, cur_depth, cur_K, rel_T[:3, :3], rel_T[:3, 3])
        # img_list.append({'img': cur_img, 'title': str(0)})
        # wrap_list.append({'img': next2cur, 'title': str(0+1) + 'to' + str(0)})
        # img_list.append({'img': next_img, 'title': str(1)})
        # img_list.append({'img': cur_depth, 'title': 'depth' + str(0)})
        #
        # show_multiple_img(img_list + wrap_list, title='dataset debug', num_cols=2)

        # I, d, sel_indices, K, T, T_gt, q, t, q_gt, t_gt = lstm_preprocess(seq_dict, num_pyramid=3, M=2000, add_noise_func=add_drift_noise,
        #                                                                   rot_noise_deg=8.0, displacement_dist_std=0.08)
        # Tcw = seq_dict['Tcw']
        # K = seq_dict['K']
        # I = seq_dict['img']
        # d = seq_dict['depth']
        # Tcw_n = Tcw[0].numpy()
        # anchor_Tcw = seq_dict['anchor_Tcw'][0].numpy()
        # # Tcw_n[0] = anchor_Tcw
        # fig = plt.figure()
        # ax = plt.gca()
        # plot_array_seq_2d(Tcw_n, plt_axes=ax, color=(0, 0, 1), show_view_direction=True, legend='GT')
        #
        # R, t = Rt(anchor_Tcw)
        # C = camera_center_from_Tcw(R, t)
        # view_direction = np.dot(R.T, np.asarray([0, 0, 1])) * 0.5
        # ax.arrow(C[0], C[2], view_direction[0], view_direction[2], width=0.05,
        #          head_width=0.3, head_length=0.2, fc='k', color='r')
        # plt.show()

        input('wait')
