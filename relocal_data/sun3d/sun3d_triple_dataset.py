import numpy as np
import pickle
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader

from frame_seq_data import K_from_frame
from seq_data.sun3d.read_util import read_sun3d_depth
from visualizer.visualizer_2d import show_multiple_img


class Sun3DTripleDataset(Dataset):

    def __init__(self, base_dir, triple_list, transform=None, output_dim=(3, 240, 320), sel_sample_num=5):
        super(Sun3DTripleDataset, self).__init__()

        self.output_dim = output_dim
        self.sel_sample_num = sel_sample_num

        # pre-defined transformation
        self.transform_func = transform
        self.base_dir = base_dir
        self.triple_list = triple_list


    def __len__(self):
        return len(self.triple_list)

    def __getitem__(self, item):

        C, H, W = self.output_dim

        triple = self.triple_list[item]
        anchor_frame = triple['anchor']
        pos_frames = triple['positive']
        neg_frames = triple['negative']
        sel_idces = np.random.choice(len(pos_frames), self.sel_sample_num, replace=False)
        pos_frames = [pos_frames[i] for i in sel_idces]
        neg_frames = [neg_frames[i] for i in sel_idces]

        data_dict = {}
        pos_dict = {'img': [], 'depth': [], 'Tcw': [], 'K': [], 'ori_img': []}
        neg_dict = {'img': [], 'depth': [], 'Tcw': [], 'K': [], 'ori_img': []}
        for i, frame in enumerate([anchor_frame] + pos_frames + neg_frames):
            K = K_from_frame(frame)
            Tcw = np.asarray(frame['extrinsic_Tcw'], dtype=np.float32).reshape((3, 4))
            img_file_name = frame['file_name']
            depth_file_name = frame['depth_file_name']

            # Load image
            img = cv2.imread(os.path.join(self.base_dir, img_file_name))
            if img is None:
                raise Exception('Can not load image:%s' % img_file_name)
            ori_H, ori_W, _ = img.shape
            img = cv2.cvtColor(cv2.resize(img, dsize=(W, H)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            # Load the depth map
            depth = read_sun3d_depth(os.path.join(self.base_dir, depth_file_name))
            depth = cv2.resize(depth, dsize=(W, H), interpolation=cv2.INTER_NEAREST)
            depth[depth < 1e-5] = 1e-5

            # convert to torch.tensor
            ori_img_tensor = torch.from_numpy(img.transpose((2, 0, 1)))  # (C, H, W)
            img_tensor = ori_img_tensor.clone()
            if self.transform_func:
                img_tensor = self.transform_func(img_tensor)
            depth_tensor = torch.from_numpy(depth).view(1, H, W)  # (1, H, W)
            K[0, 0] *= W / ori_W
            K[0, 2] *= W / ori_W
            K[1, 1] *= H / ori_H
            K[1, 2] *= H / ori_H
            K_tensor = torch.from_numpy(K)  # (3, 3)
            Tcw_tensor = torch.from_numpy(Tcw)  # (3, 4)

            if i == 0:
                data_dict['anchor_img'] = img_tensor
                data_dict['anchor_depth'] = depth_tensor
                data_dict['anchor_Tcw'] = Tcw_tensor
                data_dict['anchor_K'] = K_tensor
                data_dict['anchor_ori_img'] = ori_img_tensor
            elif i < len(pos_frames) + 1:
                pos_dict['img'].append(img_tensor)
                pos_dict['depth'].append(depth_tensor)
                pos_dict['Tcw'].append(Tcw_tensor)
                pos_dict['K'].append(K_tensor)
                pos_dict['ori_img'].append(ori_img_tensor)
            else:
                neg_dict['img'].append(img_tensor)
                neg_dict['depth'].append(depth_tensor)
                neg_dict['Tcw'].append(Tcw_tensor)
                neg_dict['K'].append(K_tensor)
                neg_dict['ori_img'].append(ori_img_tensor)

        pos_dict['img'] = torch.stack(pos_dict['img'], dim=0)       # (pos_num, C, H, W)
        pos_dict['depth'] = torch.stack(pos_dict['depth'], dim=0)   # (pos_num, 1, H, W)
        pos_dict['Tcw'] = torch.stack(pos_dict['Tcw'], dim=0)       # (pos_num, 3, 4)
        pos_dict['K'] = torch.stack(pos_dict['K'], dim=0)           # (pos_num, 3, 3)
        pos_dict['ori_img'] = torch.stack(pos_dict['ori_img'], dim=0)       # (pos_num, C, H, W)
        data_dict['pos_img'] = pos_dict['img']
        data_dict['pos_depth'] = pos_dict['depth']
        data_dict['pos_Tcw'] = pos_dict['Tcw']
        data_dict['pos_K'] = pos_dict['K']
        data_dict['pos_ori_img'] = pos_dict['ori_img']

        neg_dict['img'] = torch.stack(neg_dict['img'], dim=0)       # (neg_num, C, H, W)
        neg_dict['depth'] = torch.stack(neg_dict['depth'], dim=0)   # (neg_num, 1, H, W)
        neg_dict['Tcw'] = torch.stack(neg_dict['Tcw'], dim=0)       # (neg_num, 3, 4)
        neg_dict['K'] = torch.stack(neg_dict['K'], dim=0)           # (neg_num, 3, 3)
        neg_dict['ori_img'] = torch.stack(neg_dict['ori_img'], dim=0)  # (pos_num, C, H, W)
        data_dict['neg_img'] = neg_dict['img']
        data_dict['neg_depth'] = neg_dict['depth']
        data_dict['neg_Tcw'] = neg_dict['Tcw']
        data_dict['neg_K'] = neg_dict['K']
        data_dict['neg_ori_img'] = neg_dict['ori_img']

        return data_dict


if __name__ == '__main__':
    base_dir = '/mnt/Exp_2/SUN3D/'
    list_file = '/mnt/Exp_2/SUN3D/relocal_train.bin'
    with open(list_file, 'rb') as f:
        item_list = pickle.load(f, encoding='latin1')
    data_set = Sun3DTripleDataset(base_dir, item_list, transform=None, output_dim=(3, 480, 640))
    data_loader = DataLoader(data_set, batch_size=2, num_workers=2, shuffle=True)
    print('size of the dataset:', len(data_set))

    for data_dict in data_loader:
        print(data_dict['pos_img'].shape)

        K = data_dict['anchor_K']

        anchor_img = data_dict['anchor_img']
        anchor_ori_img = data_dict['anchor_ori_img']
        anchor_depth = data_dict['anchor_depth']
        anchor_Tcw = data_dict['anchor_Tcw']

        pos_img = data_dict['pos_img']
        pos_ori_img = data_dict['pos_ori_img']
        pos_depth = data_dict['pos_depth']
        pos_Tcw = data_dict['pos_Tcw']

        neg_img = data_dict['neg_img']
        neg_ori_img = data_dict['neg_ori_img']
        neg_depth = data_dict['neg_depth']
        neg_Tcw = data_dict['neg_Tcw']

        sel_idces = np.random.choice(5, 3, replace=False)

        show_multiple_img([{'img': anchor_ori_img[0].numpy().transpose((1, 2, 0)), 'title': 'anchor_ori_img'},
                           {'img': anchor_depth[0].numpy()[0], 'title': 'anchor_depth', 'cmap': 'jet'},
                           {'img': anchor_img[0].numpy().transpose((1, 2, 0)), 'title': 'anchor_img'},

                           {'img': pos_ori_img[0, sel_idces[0]].numpy().transpose((1, 2, 0)), 'title': 'pos_ori_img0'},
                           {'img': pos_ori_img[0, sel_idces[1]].numpy().transpose((1, 2, 0)), 'title': 'pos_ori_img1'},
                           {'img': pos_ori_img[0, sel_idces[2]].numpy().transpose((1, 2, 0)), 'title': 'pos_ori_img2'},

                           {'img': neg_ori_img[0, sel_idces[0]].numpy().transpose((1, 2, 0)), 'title': 'neg_ori_img0'},
                           {'img': neg_ori_img[0, sel_idces[1]].numpy().transpose((1, 2, 0)), 'title': 'neg_ori_img1'},
                           {'img': neg_ori_img[0, sel_idces[2]].numpy().transpose((1, 2, 0)), 'title': 'neg_ori_img2'}

                           ],title='Dataset Debug', num_cols=3)

        input('wait')
