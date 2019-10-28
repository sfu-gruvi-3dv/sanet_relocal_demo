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


class SUN3DEmbedSceneDataset(Dataset):

    def __init__(self, seq_data_list, base_dir, embed_dir, transform=None, rand_flip=False, output_dim=(3, 240, 320), supervised_tag='train_anchor_frames'):
        super(SUN3DEmbedSceneDataset, self).__init__()
        self.seq_list = seq_data_list
        self.base_dir = base_dir
        self.embed_dir = embed_dir
        self.random_flip = rand_flip
        self.transform_func = transform
        self.output_dim = output_dim
        self.supervised_out_tag = supervised_tag
        self.cache_embed_buffer = False
        self.embed_cache = {}

    def __len__(self):
        return len(self.seq_list)

    def load_frame_2_tensors(self, frame, out_frame_dim):
        C, H, W = out_frame_dim
        K = K_from_frame(frame)
        Tcw = np.asarray(frame['extrinsic_Tcw'], dtype=np.float32).reshape((3, 4))
        Rcw, tcw = Tcw[:3, :3], Tcw[:3, 3]
        img_file_name = frame['file_name']
        depth_file_name = frame['depth_file_name']

        # Load image
        img = cv2.imread(os.path.join(self.base_dir, img_file_name))
        ori_H, ori_W, _ = img.shape
        img = cv2.cvtColor(cv2.resize(img, dsize=(W, H)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Load the depth map:
        # depth = read_sun3d_depth(os.path.join(self.base_dir, depth_file_name))
        # depth = cv2.resize(depth, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        # depth[depth < 1e-5] = 1e-5

        # camera intrinsic parameters:
        # K[0, 0] *= W / ori_W
        # K[0, 2] *= W / ori_W
        # K[1, 1] *= H / ori_H
        # K[1, 2] *= H / ori_H
        # K_tensor = torch.from_numpy(K)  # (3, 3)

        # camera motion representation: (center, rotation_center2world)
        c = camera_center_from_Tcw(Rcw, tcw)
        Rwc = np.eye(4)
        Rwc[:3, :3] = Rcw.T
        q = quaternion_from_matrix(Rwc)
        log_q = log_quat(q)
        pose_vector = np.concatenate((c, log_q)).astype(np.float32)

        # convert to torch.tensor
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1)))  # (C, H, W)
        if self.transform_func:
            img_tensor = self.transform_func(img_tensor)
        # depth_tensor = torch.from_numpy(depth).view(1, H, W)  # (1, H, W)

        pose_vector = torch.from_numpy(pose_vector)     # (1, 3)

        return pose_vector, img_tensor

    def __getitem__(self, idx):
        input_frames = self.seq_list[idx]['sub_frames'].frames
        C, H, W = self.output_dim

        rand_flip_flag = np.random.randint(2) if self.random_flip else 0
        if rand_flip_flag == 1:
            input_frames = input_frames[::-1]

        # Extract embedding
        seq_name = SceneEmbedSet.extract_seq_name(input_frames[0]['file_name'])
        if seq_name not in self.embed_cache:
            seq_cache = SceneEmbedSet(self.embed_dir, seq_name)
            if self.cache_embed_buffer:
                self.embed_cache[seq_name] = seq_cache
        else:
            seq_cache = self.embed_cache[seq_name]

        frame_names = [frame['file_name'] for frame in input_frames]
        n_frames = len(frame_names)
        frame_embeds = torch.from_numpy(seq_cache.extract_embed(frame_names))

        # Sample a item
        sample = random.choice(self.seq_list[idx][self.supervised_out_tag])
        pose_vector, img_tensor = self.load_frame_2_tensors(sample, self.output_dim)

        return {'in_scene_feat': frame_embeds, 'img': img_tensor, 'pose': pose_vector}


if __name__ == '__main__':
    with open('/local-scratch/SUN3D_Valid/reloc_subseq_with_anchor_valid.bin', 'rb') as f:
        data_list = pickle.load(f)
    data_set = SUN3DEmbedSceneDataset(seq_data_list=data_list,
                                      embed_dir='/local-scratch3/reloc_logs/embed',
                                      base_dir='/local-scratch/SUN3D_Valid', transform=None, rand_flip=False)
    # sample = data_set.__getitem__(12)
    data_loader = DataLoader(data_set, batch_size=1, num_workers=0, shuffle=True)
    print('size of the dataset:', len(data_set))

    for seq_dict in data_loader:
        pose_vector = seq_dict['pose']
        I = seq_dict['img']
        in_feat = seq_dict['in_scene_feat']

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
