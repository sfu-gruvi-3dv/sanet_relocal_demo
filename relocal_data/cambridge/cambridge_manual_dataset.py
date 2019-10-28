import numpy as np
import pickle, cv2, random, os, torch

from collections import namedtuple
from visualizer.visualizer_2d import show_multiple_img
from torch.utils.data import Dataset, DataLoader
from skimage import exposure

from libs.pycbf_filter.depth_fill import fill_depth_cross_bf
from img_proc.img_dim import crop_by_intrinsic
from core_3dv.camera_operator import *
from core_3dv.camera_operator_gpu import x_2d_coords
from frame_seq_data import K_from_frame, FrameSeqData
from core_math.transfom import quaternion_from_matrix
from seq_data.seven_scenes.read_util import *
from relocal_data.seven_scene.seven_scene_dict_preprocess import preprocess
from core_io.depth_io import load_depth_from_tiff

class CambridgeManualDataset(Dataset):

    def __init__(self, base_dir, seq_frame_list, transform=None, output_dim=(3, 240, 320), nsample_per_group=5, adjust_gamma=False):
        super(CambridgeManualDataset, self).__init__()

        self.base_dir = base_dir
        self.transform_func = transform
        self.output_dim = output_dim
        self.seq_frame_list = seq_frame_list
        self.nsample_per_group = nsample_per_group
        self.adjust_gamma = adjust_gamma

    def __len__(self):
        return len(self.seq_frame_list) // self.nsample_per_group

    def size_of_frames(self):
        return len(self.seq_frame_list)

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
        img = cv2.imread(os.path.join(self.base_dir, img_file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.adjust_gamma:
            def enhance_equal_hist(img, rgb=False):
                if rgb is True:
                    for channel in range(img.shape[2]):  # equalizing each channel
                        img[:, :, channel] = exposure.equalize_hist(img[:, :, channel])
                else:
                    img = exposure.equalize_hist(img)
                return img.astype(np.float32)
            img = enhance_equal_hist(img, rgb=True)
        depth = load_depth_from_tiff((os.path.join(self.base_dir, depth_file_name)))
        ori_H, ori_W, _ = img.shape

        # Post-process image and depth (fill the holes with cross bilateral filter)
        img = cv2.resize(img, dsize=(int(W), int(H)))
        depth = cv2.resize(depth, dsize=(int(W), int(H)), interpolation=cv2.INTER_NEAREST)
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
        input_frames = self.seq_frame_list[idx * self.nsample_per_group : (idx+1) * self.nsample_per_group]

        C, H, W = self.output_dim

        # Load frames
        frame_name_list = []
        pose_vectors = []
        img_tensors = []
        depth_tensors = []
        K_tensors = []
        Tcw_tensors = []
        ori_img_tensors = []
        for frame in input_frames:
            pose_vector, img_tensor, depth_tensor, K_tensor, Tcw_tensor, ori_img_tensor, _ = self.load_frame_2_tensors(frame, self.output_dim)
            frame_name_list.append(frame['file_name'])
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

        return {'frames_names': frame_name_list,
                'frames_img': img_tensors, 'frames_depth': depth_tensors, 'frames_pose': pose_vectors,
                'frames_K': K_tensors, 'frames_Tcw': Tcw_tensors, 'frames_ori_img': ori_img_tensors}

if __name__ == '__main__':

    # read frames from binary file
    frame_bin_path = '/home/luwei/mnt/Exp_1/7scenes/bins/redkitchen_skip25_seq-03.bin'
    with open(frame_bin_path, 'rb') as f:
        frames = pickle.load(f)
        frames = frames[0]['sub_frames'].frames

    scene_set = CambridgeManualDataset(base_dir='/home/luwei/mnt/Exp_1/7scenes',
                                        seq_frame_list=frames,
                                        transform=None,
                                        fill_depth_holes=True,
                                        output_dim=(3, 192, 256))
    data_loader = DataLoader(scene_set, batch_size=1, num_workers=0, shuffle=False)
    print('size of the dataset:', len(scene_set))

    with torch.cuda.device(1):
        x_2d = x_2d_coords(192, 256, scene_set.size_of_frames()).view((scene_set.size_of_frames(), -1, 2)).cuda()

        seq_dict = next(iter(data_loader))
        res = preprocess(x_2d,
                         seq_dict['frames_img'].cuda(),
                         seq_dict['frames_depth'].cuda(),
                         seq_dict['frames_K'].cuda(),
                         seq_dict['frames_Tcw'].cuda(0),
                         scene_ori_rgb=seq_dict['frames_ori_img'].cuda())
        scene_input, scene_ori_rgb, X_world, valid_map, scene_center = res

        print(scene_input.shape)
        print(scene_ori_rgb.shape)
        print(X_world.shape)
        print(valid_map.shape)
        print(scene_center.shape)

    # for seq_dict in data_loader:
    #
    #     query_img = seq_dict['frames_img']  # depth[0 + 1].reshape((256, 256, 1))
    #     print(query_img.shape)
    #     query_depth = seq_dict['frames_depth'][0].numpy()
    #     query_Tcw = seq_dict['frames_pose'][0].numpy()
    #     show_multiple_img([{'img': query_img, 'title': 'query rgb'},
    #                        {'img': query_depth, 'title': 'query depth'}],
    #                       title='dataset debug', num_cols=2)

    input('wait')
