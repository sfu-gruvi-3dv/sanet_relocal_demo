import os, sys, cv2, json, pickle, h5py, random
import numpy as np
import matplotlib.pyplot as plt
from core_math.transfom import quaternion_from_matrix
from torch.utils.data import Dataset, DataLoader
from core_3dv.camera_operator import *
from frame_seq_data import K_from_frame, FrameSeqData
from visualizer.visualizer_2d import show_multiple_img
from core_io.depth_io import load_depth_from_tiff
from skimage.exposure import adjust_gamma

def clamp_data_with_ratio(map, ratio=0.95, fill_value=1e-5):
    thres = np.percentile(map.ravel(), q=ratio*100)
    map[np.where(map > thres)] = fill_value
    return map

class CambridgeDataset(Dataset):
    """ Dataset of SUN3D and SCENENN
    """
    def __init__(self, seq_data_list, cambridge_base_dir, transform=None, remove_depth_outlier_ratio=0.98, rand_flip=False, random_gamma=True, output_dim=(3, 240, 320)):
        super(CambridgeDataset, self).__init__()
        self.seq_list = seq_data_list
        self.cambridge_base_dir = cambridge_base_dir
        self.random_flip = rand_flip
        self.transform_func = transform
        self.remove_depth_outlier = remove_depth_outlier_ratio
        self.output_dim = output_dim
        self.random_gamma = random_gamma
        self.random_gamma_thres = (0.3, 1.2)
        self.supervised_out_tag = 'train_anchor_frames'

    def __len__(self):
        return len(self.seq_list)

    def load_frame_2_tensors(self, frame, out_frame_dim):
        C, H, W = out_frame_dim

        tag = frame['tag']
        if tag is not None:
            is_neg = True if 'n' in tag else False
        else:
            is_neg = False

        K = K_from_frame(frame)
        Tcw = np.asarray(frame['extrinsic_Tcw'][:3, :], dtype=np.float32).reshape((3, 4))
        Rcw, tcw = Tcw[:3, :3], Tcw[:3, 3]
        img_file_name = frame['file_name']
        depth_file_name = frame['depth_file_name']

        # Load image and depth
        img = cv2.imread(os.path.join(self.cambridge_base_dir, img_file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        depth = load_depth_from_tiff((os.path.join(self.cambridge_base_dir, depth_file_name)))
        ori_H, ori_W, _ = img.shape

        # Post-process image and depth (fill the holes with cross bilateral filter)
        img = cv2.resize(img, dsize=(int(W), int(H)))
        if self.random_gamma:
            gamma = np.random.uniform(low=self.random_gamma_thres[0], high=self.random_gamma_thres[1])
            img = adjust_gamma(img, gamma)

        depth = cv2.resize(depth, dsize=(int(W), int(H)), interpolation=cv2.INTER_NEAREST)
        if self.remove_depth_outlier > 0:
            depth = clamp_data_with_ratio(depth, ratio=self.remove_depth_outlier, fill_value=1e-5)
        depth[depth < 1e-5] = 1e-5

        # camera intrinsic parameters:
        K[0, 0] *= (W/ori_W)
        K[0, 2] *= (W/ori_W)
        K[1, 1] *= (H/ori_H)
        K[1, 2] *= (H/ori_H)

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
        pose_vector = torch.from_numpy(pose_vector)           # (1, 3)
        Tcw_tensor = torch.from_numpy(Tcw)                    # (3, 4)
        K_tensor = torch.from_numpy(K)                        # (3, 3)
        neg_tensor = torch.from_numpy(np.asarray([1], dtype=np.int32)) if is_neg is True else \
            torch.from_numpy(np.asarray([0], dtype=np.int32))

        return pose_vector, img_tensor, depth_tensor, K_tensor, Tcw_tensor, ori_img_tensor, neg_tensor

    def __getitem__(self, idx):
        input_frames = self.seq_list[idx]['sub_frames'].frames
        random.shuffle(input_frames)
        C, H, W = self.output_dim

        # load frames
        pose_vectors = []
        img_tensors = []
        depth_tensors = []
        K_tensors = []
        Tcw_tensors = []
        ori_img_tensors = []
        neg_tag_tensors = []
        for frame in input_frames:
            pose_vector, img_tensor, depth_tensor, K_tensor, Tcw_tensor, ori_img_tensor, neg_tag_tensor \
                = self.load_frame_2_tensors(frame, self.output_dim)
            pose_vectors.append(pose_vector)
            img_tensors.append(img_tensor)
            depth_tensors.append(depth_tensor)
            K_tensors.append(K_tensor)
            Tcw_tensors.append(Tcw_tensor)
            ori_img_tensors.append(ori_img_tensor)
            neg_tag_tensors.append(neg_tag_tensor)

        pose_vectors = torch.stack(pose_vectors, dim=0)
        img_tensors = torch.stack(img_tensors, dim=0)
        depth_tensors = torch.stack(depth_tensors, dim=0)
        K_tensors = torch.stack(K_tensors, dim=0)
        Tcw_tensors = torch.stack(Tcw_tensors, dim=0)
        ori_img_tensors = torch.stack(ori_img_tensors, dim=0)
        neg_tag_tensors = torch.stack(neg_tag_tensors, dim=0).view(len(input_frames))

        # Sample a item
        sample = random.choice(self.seq_list[idx][self.supervised_out_tag])
        pose_vector, img_tensor, depth_tensor, K_tensor, Tcw_tensor, ori_img_tensor, _ = \
            self.load_frame_2_tensors(sample, self.output_dim)

        return {'frames_img': img_tensors,
                'frames_depth': depth_tensors,
                'frames_pose': pose_vectors,
                'frames_K': K_tensors,
                'frames_Tcw': Tcw_tensors,
                'frames_ori_img': ori_img_tensors,
                'frames_neg_tags': neg_tag_tensors,
                'img': img_tensor,
                'depth': depth_tensor,
                'pose': pose_vector,
                'K': K_tensor,
                'Tcw': Tcw_tensor,
                'ori_img': ori_img_tensor}


if __name__ == '__main__':

    with open('/mnt/Exp_3/cambridge/StMarysChurch/train.bin', 'rb') as f:
        data_list = pickle.load(f)

    data_set = CambridgeDataset(seq_data_list=data_list,
                                cambridge_base_dir='/mnt/Exp_3/cambridge',
                                transform=None, rand_flip=False, random_gamma=True,
                                output_dim=(3, 192, 256))

    # sample = data_set.__getitem__(12)
    data_loader = DataLoader(data_set, batch_size=1, num_workers=0, shuffle=False)
    print('size of the dataset:', len(data_set))

    for seq_dict in data_loader:
        Tcw = seq_dict['frames_Tcw'][0].numpy()
        I = seq_dict['frames_img'][0].numpy().transpose(0, 2, 3, 1)
        depth = seq_dict['frames_depth'][0, :, 0].numpy()
        K = seq_dict['frames_K'][0].numpy()
        neg_tags = seq_dict['frames_neg_tags'][0].numpy()
        print(neg_tags)

        query_img = seq_dict['img'][0].numpy().transpose(1, 2, 0)  # depth[0 + 1].reshape((256, 256, 1))
        query_depth = seq_dict['depth'][0, 0].numpy()
        query_Tcw = seq_dict['Tcw'][0].numpy()

        img_list = []
        wrap_list = []
        for i in range(min(I.shape[0] - 1, 5)):
            cur_img = I[i]
            next_img = I[i+1]
            cur_K = K[i]
            neg_tag = 'N' if neg_tags[i] == 1 else 'P'
            rel_T = relateive_pose(Tcw[i, :3, :3], Tcw[i, :3, 3], Tcw[i+1, :3, :3], Tcw[i+1, :3, 3])
            next2cur, _ = wrapping(cur_img, next_img, depth[i], cur_K, rel_T[:3, :3], rel_T[:3, 3])
            img_list.append({'img': cur_img, 'title': str(i) + neg_tag})
            wrap_list.append({'img': next2cur, 'title': str(i) + ' to ' + str(i+1)})
            if i == min(I.shape[0] - 1, 5) - 1:
                neg_tag = 'N' if neg_tags[i+1] == 1 else 'P'
                img_list.append({'img': next_img, 'title': str(i+1) + neg_tag})

        wrap_list.append({'img': query_img, 'title': 'query'})
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
