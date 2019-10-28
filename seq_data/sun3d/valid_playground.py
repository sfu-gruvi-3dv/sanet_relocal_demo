import pickle
import torch
import cv2
import core_3dv.camera_operator as cam_opt
from visualizer.visualizer_3d import Visualizer
from visualizer.visualizer_2d import show_multiple_img
from seq_data.sun3d.sun3d_seq_dataset import Sun3DSeqDataset
from banet_track.ba_module import *
from seq_data.seq_preprocess import add_drift_noise, add_gaussian_noise
import lstm.quaternion_module as quat_mod
from lstm.lstm_module import wrap
from seq_data.plot_seq_2d import plot_array_seq_2d
import os
from tqdm import tqdm

show_3d_traj = False

with open('/mnt/Exp_2/SUN3D_Valid/subseq.bin', 'rb') as f:
    data_list = pickle.load(f)
data_set = Sun3DSeqDataset(seq_data_list=data_list, base_dir='/mnt/Exp_2/SUN3D_Valid', transform=None, rand_flip=False)

# Output
out_dir = '/mnt/Exp_3/valid_dataset_dump'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

in_dir = os.path.join('/mnt/Tango/lstm_reg_train', 'lstm_dump_pose')

n_samples = len(data_set)
for sample_idx in tqdm(range(0, 20, 1)):
    sample = data_set.__getitem__(sample_idx)

    # print(sample['Tcw'])
    # print(sample['K'])

    Tcw = sample['Tcw']
    K = sample['K']
    I = sample['img']
    d = sample['depth']

    L = Tcw.shape[0]
    C, H, W = I.shape[1:]

    T_noise, disp_sigmaq = add_drift_noise(Tcw.cpu().numpy(), rot_noise_deg=5.0, displacement_dist_std=0.05)
    # print('Displacement:', disp_sigmaq)
    T_noise = torch.from_numpy(T_noise)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    fig = plt.figure()
    ax = plt.gca()
    Tcw_n = np.load(os.path.join(in_dir, "Tcw_%05d.npy") % sample_idx)
    T_noise_n = np.load(os.path.join(in_dir, "Tcw_n%05d.npy") % sample_idx)
    T_pred_n = np.load(os.path.join(in_dir, "Tcw_p%05d.npy") % sample_idx)

    # T_noise_n = T_noise.cpu().numpy().reshape(L, 3, 4)
    for f_idx in range(0, L):
        T_item = Tcw_n[f_idx, :, :]
        T_n_item = T_noise_n[f_idx, :, :]
        c1 = cam_opt.camera_center_from_Tcw(T_item[:3, :3], T_item[:3, 3])
        c2 = cam_opt.camera_center_from_Tcw(T_n_item[:3, :3], T_n_item[:3, 3])
        print('Baseline: ', np.linalg.norm(c1 -c2))

    plot_array_seq_2d(Tcw_n, plt_axes=ax, color=(0, 0, 1), show_view_direction=True, legend='GT')
    plot_array_seq_2d(T_noise_n, plt_axes=ax, color=(1, 0, 0), show_view_direction=True, legend='Noise')
    plot_array_seq_2d(T_pred_n, plt_axes=ax, color=(0, 1, 0), show_view_direction=True, legend='Pred')

    # plt.show()
    plt.savefig(os.path.join(out_dir, "%05d_path.png" % sample_idx))
    plt.clf()

    if show_3d_traj:
        #      cur_Tcw = Tcw[0]
        # RGB: cur_Image = I[0].permute(1, 2, 0)
        # D:   cur_depth = d[0].squeeze(0)
        t = torch.Tensor()
        vis = Visualizer()

        d = d.cuda()
        K = K.cuda()
        Tcw = Tcw.cuda()
        Twc = batched_mat_inv(transform_mat44(Tcw))
        x_2d = x_2d_coords_torch(L, H, W)
        X_3d_a = batched_pi_inv(K, x_2d.view(L, H * W, 2), d.view(L, H * W, 1))
        X_3d_w = batched_transpose(Twc[:, :3, :3], Twc[:, :3, 3], X_3d_a)

        frame_idx = 0
        def keyPressEvent(obj, event):
            global frame_idx
            key = obj.GetKeySym()
            if key == 'Right':
                cur_n_Tcw = T_noise[frame_idx]
                cur_Tcw = Tcw[frame_idx]
                cur_image = I[frame_idx].permute(1, 2, 0).cpu().numpy()
                X_3d = X_3d_w[frame_idx].cpu().numpy()

                vis.set_point_cloud(X_3d, cur_image.reshape((H*W, 3)))
                vis.add_frame_pose(cur_Tcw[:3, :3].cpu().numpy(), cur_Tcw[:3, 3].cpu().numpy(), camera_obj_scale=0.05)
                vis.add_frame_pose(cur_n_Tcw[:3, :3].cpu().numpy(), cur_n_Tcw[:3, 3].cpu().numpy(), color=(0.5, 0, 0), camera_obj_scale=0.05)
                frame_idx += 1

        vis.bind_keyboard_event(keyPressEvent)
        vis.show()
    else:
        # Show Wrapped image
        img_list = []

        I = I.cuda()
        d = d.cuda()
        K = K.cuda()
        # Tcw = Tcw.cuda()
        Tcw = torch.from_numpy(Tcw_n).view(L, 3, 4).cuda()
        T_noise = torch.from_numpy(T_noise_n).view(L, 3, 4).cuda()
        T_pred = torch.from_numpy(T_pred_n).view(L, 3, 4).cuda()

        I = I.view(L, 1, C, H, W)
        d = d.view(L, 1, 1, H, W)
        rel_T = batched_relative_pose_mat44(Tcw[:-1, :3, :3], Tcw[:-1, :3, 3], Tcw[1:, :3, :3], Tcw[1:, :3, 3])
        rel_T = rel_T.view(L-1, 1, 4, 4)
        n_rel_T = batched_relative_pose_mat44(T_noise[:-1, :3, :3], T_noise[:-1, :3, 3], T_noise[1:, :3, :3], T_noise[1:, :3, 3])
        n_rel_T = n_rel_T.view(L-1, 1, 4, 4)
        p_rel_T = batched_relative_pose_mat44(T_pred[:-1, :3, :3], T_pred[:-1, :3, 3], T_pred[1:, :3, :3], T_pred[1:, :3, 3])
        p_rel_T = p_rel_T.view(L-1, 1, 4, 4)

        K = K.view(L, 1, 3, 3)
        pre_x_2d = x_2d_coords_torch((L - 1) * 1, H, W).cuda()
        I_wrap, d_wrap = wrap(I, d, rel_T, K, pre_cached_x_2d=pre_x_2d)
        n_I_wrap, n_d_wrap = wrap(I, d, n_rel_T, K, pre_cached_x_2d=pre_x_2d)
        p_I_wrap, p_d_wrap = wrap(I, d, p_rel_T, K, pre_cached_x_2d=pre_x_2d)

        for img_idx in range(0, L-1):
            img = I[img_idx, 0, :, :, :].permute(1, 2, 0).cpu().numpy()
            img_b2a = I_wrap[img_idx, 0, :, :, :].permute(1, 2, 0).cpu().numpy()
            n_img_b2a = n_I_wrap[img_idx, 0, :, :, :].permute(1, 2, 0).cpu().numpy()
            p_img_b2a = p_I_wrap[img_idx, 0, :, :, :].permute(1, 2, 0).cpu().numpy()
            img_b = I[img_idx+1, 0, :, :, :].permute(1, 2, 0).cpu().numpy()
            img_list.append({'img':img, 'title': 'F' + str(img_idx)})
            img_list.append({'img':img_b2a, 'title': 'B_to_A' + str(img_idx)})
            img_list.append({'img':p_img_b2a, 'title': 'Pred B_to_A' + str(img_idx)})
            img_list.append({'img':n_img_b2a, 'title': 'Noise B_to_A' + str(img_idx)})
            img_list.append({'img':img_b, 'title': 'F' + str(img_idx+1)})

        show_multiple_img(img_list, title='Preview', num_cols=5, figsize=(8, 26), show=False)
        plt.savefig(os.path.join(out_dir, "%05d_sample.png" % sample_idx))