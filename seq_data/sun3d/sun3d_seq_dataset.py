import numpy as np
import os
import sys
import cv2
import json
import torch
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from core_3dv.camera_operator import *
from frame_seq_data import K_from_frame, FrameSeqData
from seq_data.sun3d.read_util import read_sun3d_depth
from seq_data.seq_preprocess import add_drift_noise
from dataset.preprocess import lstm_preprocess
from seq_data.plot_seq_2d import plot_array_seq_2d


class Sun3DSeqDataset(Dataset):

    def __init__(self, seq_data_list, base_dir, transform=None, rand_flip=False, output_dim=(3, 240, 320)):
        super(Sun3DSeqDataset, self).__init__()
        self.seq_list = seq_data_list
        self.base_dir = base_dir
        self.random_flip = rand_flip
        self.transform_func = transform
        self.output_dim = output_dim

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, idx):
        frames = self.seq_list[idx].frames

        C, H, W = self.output_dim

        rand_flip_flag = np.random.randint(2) if self.random_flip else 0
        if rand_flip_flag == 0:
            # sequence order not changed
            pass
        else:
            # sequence order reversed
            frames = frames[::-1]

        # Read frames
        img_tensors = []
        depth_tensors = []
        K_tensors = []
        Tcw_tensors = []

        for frame in frames:
            K = K_from_frame(frame)
            Tcw = np.asarray(frame['extrinsic_Tcw'], dtype=np.float32).reshape((3, 4))
            img_file_name = frame['file_name']
            depth_file_name = frame['depth_file_name']

            # Load image
            img = cv2.imread(os.path.join(self.base_dir, img_file_name))
            ori_H, ori_W, _ = img.shape
            img = cv2.cvtColor(cv2.resize(img, dsize=(W, H)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Load the depth map
            depth = read_sun3d_depth(os.path.join(self.base_dir, depth_file_name))
            depth = cv2.resize(depth, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
            depth[depth < 1e-5] = 1e-5

            # convert to torch.tensor
            img_tensor = torch.from_numpy(img.transpose((2, 0, 1)))                         # (C, H, W)
            if self.transform_func:
                img_tensor = self.transform_func(img_tensor)
            depth_tensor = torch.from_numpy(depth).view(1, H, W)                            # (1, H, W)
            img_tensors.append(img_tensor)
            depth_tensors.append(depth_tensor)
            K[0, 0] *= W / ori_W
            K[0, 2] *= W / ori_W
            K[1, 1] *= H / ori_H
            K[1, 2] *= H / ori_H

            K_tensor = torch.from_numpy(K)                                                  # (3, 3)
            K_tensors.append(K_tensor)
            Tcw_tensor = torch.from_numpy(Tcw)                                              # (3, 4)
            Tcw_tensors.append(Tcw_tensor)

        img_tensors = torch.stack(img_tensors, dim=0)                                   # (frame_num, C, H, W)
        depth_tensors = torch.stack(depth_tensors, dim=0)                               # (frame_num, 1, H, W)
        K_tensors = torch.stack(K_tensors, dim=0)                                       # (frame_num, 3, 3)
        Tcw_tensors = torch.stack(Tcw_tensors, dim=0)                                   # (frame_num, 3, 4)

        return {'img': img_tensors, 'depth': depth_tensors, 'Tcw': Tcw_tensors, 'K': K_tensors}


if __name__ == '__main__':
    with open('/mnt/Exp_2/SUN3D_Valid/reloc_subseq.bin', 'rb') as f:
        data_list = pickle.load(f)
    data_set = Sun3DSeqDataset(seq_data_list=data_list, base_dir='/mnt/Exp_2/SUN3D_Valid', transform=None, rand_flip=False)
    sample = data_set.__getitem__(12)
    data_loader = DataLoader(data_set, batch_size=1, num_workers=0, shuffle=True)
    print('size of the dataset:', len(data_set))

    for seq_dict in data_loader:
        # I, d, sel_indices, K, T, T_gt, q, t, q_gt, t_gt = lstm_preprocess(seq_dict, num_pyramid=3, M=2000, add_noise_func=add_drift_noise,
        #                                                                   rot_noise_deg=8.0, displacement_dist_std=0.08)
        Tcw = seq_dict['Tcw']
        K = seq_dict['K']
        I = seq_dict['img']
        d = seq_dict['depth']
        Tcw_n = Tcw[0].numpy()
        fig = plt.figure()
        ax = plt.gca()
        plot_array_seq_2d(Tcw_n, plt_axes=ax, color=(0, 0, 1), show_view_direction=True, legend='GT')
        plt.show()

        input('wait')
