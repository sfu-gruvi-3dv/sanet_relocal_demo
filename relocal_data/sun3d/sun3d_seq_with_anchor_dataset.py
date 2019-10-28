import numpy as np
import os
import sys
import cv2
import json
import torch
import pickle
import h5py
import random
import matplotlib.pyplot as plt
from core_math.transfom import quaternion_from_matrix
from torch.utils.data import Dataset, DataLoader
from core_3dv.camera_operator import *
from frame_seq_data import K_from_frame, FrameSeqData
from seq_data.sun3d.read_util import read_sun3d_depth
from seq_data.seq_preprocess import add_drift_noise
from dataset.preprocess import lstm_preprocess
from seq_data.plot_seq_2d import plot_array_seq_2d
from visualizer.visualizer_2d import show_multiple_img
from libs.pycbf_filter.depth_fill import fill_depth_cross_bf
from img_proc.img_dim import crop_by_intrinsic

class SceneEmbedSet:

    embed_buffer = None

    name2idx = None

    seq_name = None

    def __init__(self, embed_dir:str, seq_name:str, embed_file_name='embed.hdf5'):
        embed_file_path = os.path.join(embed_dir, seq_name, embed_file_name)
        if not os.path.exists(embed_file_path):
            raise Exception("No embed file found.")

        self.seq_name = seq_name
        with h5py.File(embed_file_path, 'r') as f:
            names = f['names'][()]
            self.embed_buffer = f['embedding'][()].astype(np.float32)

            # create mapping from name to index
            names_list = [name.decode('ascii') for name in names]
            self.name2idx = dict(zip(names_list, list(range(0, len(names_list), 1))))

    def extract_embed(self, frame_name_list):
        seq_name = self.extract_seq_name(frame_name_list[0])
        if seq_name != self.seq_name:
            raise Exception("Seq name is not matched.")

        embed_list = []
        for frame_name in frame_name_list:
            idx = self.name2idx[frame_name]
            embed = self.embed_buffer[idx, ...].ravel()
            embed_list.append(np.expand_dims(embed, axis=0))

        embed_array = np.concatenate(embed_list, axis=0)
        return embed_array

    @staticmethod
    def extract_seq_name(frame_img_name):
        # extract sequence name from frame_img_name
        return frame_img_name[:frame_img_name.find('/', frame_img_name.find('/') + 1)]


class SUN3DSeqWithAnchorDataset(Dataset):

    def __init__(self, seq_data_list, base_dir, transform=None, rand_flip=False, fill_depth_holes=True, output_dim=(3, 240, 320), supervised_tag='train_anchor_frames'):
        super(SUN3DSeqWithAnchorDataset, self).__init__()
        self.seq_list = seq_data_list
        self.base_dir = base_dir
        self.random_flip = rand_flip
        self.transform_func = transform
        self.output_dim = output_dim
        self.supervised_out_tag = supervised_tag
        self.fill_depth_holes = fill_depth_holes

    def __len__(self):
        return len(self.seq_list)

    def load_frame_2_tensors(self, frame, out_frame_dim, fill_depth_holes=False):
        C, H, W = out_frame_dim
        K = K_from_frame(frame)
        Tcw = np.asarray(frame['extrinsic_Tcw'], dtype=np.float32).reshape((3, 4))
        Rcw, tcw = Tcw[:3, :3], Tcw[:3, 3]
        img_file_name = frame['file_name']
        depth_file_name = frame['depth_file_name']

        # Load image and depth
        img = cv2.imread(os.path.join(self.base_dir, img_file_name))
        depth = read_sun3d_depth(os.path.join(self.base_dir, depth_file_name))
        ori_H, ori_W, _ = img.shape

        # Post-process image and depth (fill the holes with cross bilateral filter)
        resize_ratio = max(H / ori_H, W / ori_W)
        img = cv2.resize(img, dsize=(int(resize_ratio * ori_W), int(resize_ratio * ori_H)))
        depth = cv2.resize(depth, dsize=(int(resize_ratio * ori_W), int(resize_ratio * ori_H)), interpolation=cv2.INTER_NEAREST)
        if fill_depth_holes:
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

        # Extract embedding
        # seq_name = SceneEmbedSet.extract_seq_name(input_frames[0]['file_name'])
        # if seq_name not in self.embed_cache:
        #     seq_cache = SceneEmbedSet(self.embed_dir, seq_name)
        #     if self.cache_embed_buffer:
        #         self.embed_cache[seq_name] = seq_cache
        # else:
        #     seq_cache = self.embed_cache[seq_name]
        #
        # frame_names = [frame['file_name'] for frame in input_frames]
        # n_frames = len(frame_names)
        # frame_embeds = torch.from_numpy(seq_cache.extract_embed(frame_names))

        # Load frames
        pose_vectors = []
        img_tensors = []
        depth_tensors = []
        K_tensors = []
        Tcw_tensors = []
        ori_img_tensors = []
        for frame in input_frames:
            pose_vector, img_tensor, depth_tensor, K_tensor, Tcw_tensor, ori_img_tensor = self.load_frame_2_tensors(frame, self.output_dim, False)
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
        pose_vector, img_tensor, depth_tensor, K_tensor, Tcw_tensor, ori_img_tensor = self.load_frame_2_tensors(sample, self.output_dim, self.fill_depth_holes)

        return {'frames_img': img_tensors, 'frames_depth': depth_tensors, 'frames_pose': pose_vectors, 'frames_K': K_tensors, 'frames_Tcw': Tcw_tensors, 'frames_ori_img': ori_img_tensors,
                'img': img_tensor, 'depth': depth_tensor, 'pose': pose_vector, 'K': K_tensor, 'Tcw': Tcw_tensor, 'ori_img': ori_img_tensor}


if __name__ == '__main__':
    with open('/mnt/Exp_2/SUN3D_Valid/reloc_subseq5_Scene2SceneOverlap0.2_Dist2.0_valid.bin', 'rb') as f:
        data_list = pickle.load(f)
    data_set = SUN3DSeqWithAnchorDataset(seq_data_list=data_list,
                                         base_dir='/mnt/Exp_2/SUN3D_Valid', transform=None, rand_flip=False,
                                         output_dim=(3, 192, 256))
    # sample = data_set.__getitem__(12)
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
        for i in range(min(I.shape[0] - 1, 5)):
            cur_img = I[i]
            next_img = I[i+1]
            cur_K = K[i]
            rel_T = relateive_pose(Tcw[i, :3, :3], Tcw[i, :3, 3], Tcw[i+1, :3, :3], Tcw[i+1, :3, 3])
            next2cur, _ = wrapping(cur_img, next_img, depth[i], cur_K, rel_T[:3, :3], rel_T[:3, 3])
            img_list.append({'img': cur_img, 'title': str(i)})
            wrap_list.append({'img': next2cur, 'title': str(i) + ' to ' + str(i+1)})
            if i == min(I.shape[0] - 1, 5) - 1:
                img_list.append({'img': next_img, 'title': str(i+1)})

        wrap_list.append({'img': query_img, 'title': 'query image'})
        show_multiple_img(img_list + wrap_list, title='dataset debug', num_cols=max(2, min(I.shape[0], 5)))

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
