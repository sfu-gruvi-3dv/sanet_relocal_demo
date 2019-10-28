import numpy as np
import pickle
import cv2
import os
import torch
from torch.utils.data import Dataset


class SUN3DDataset(Dataset):

    def __init__(self, base_dir, dataset_list_path, transform=None,  output_dim=(3, 240, 320)):

        self.output_dim = output_dim

        # pre-defined transformation
        self.transform_func = transform
        self.base_dir = base_dir

        # read the item list
        with open(dataset_list_path, 'rb') as f:
            self.seq_list = pickle.load(f, encoding='latin1')
            self.seq_list = self.seq_list[:10]

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, item):

        C, H, W = self.output_dim

        seq = self.seq_list[item]

        img_list = []
        overlap_arr = []
        baseline_dist_arr = []
        for frame_idx, frame in enumerate(seq.frames):
            img = cv2.imread(os.path.join(self.base_dir, frame['file_name']))
            img = cv2.cvtColor(cv2.resize(img, dsize=(W, H)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            img_tensor = torch.from_numpy(img.transpose((2, 0, 1)))
            if self.transform_func:
                img_tensor = self.transform_func(img_tensor)

            img_list.append(img_tensor)
            overlap_ratio = frame['pairwise_overlap']
            overlap_ratio.insert(frame_idx, 0.0)
            baseline_dist = frame['pairwise_dist']
            baseline_dist.insert(frame_idx, 0.0)

            overlap_arr.append(np.asarray(overlap_ratio))
            baseline_dist_arr.append(np.asarray(baseline_dist))

        img_tensors = torch.stack(img_list, dim=0)
        overlap_arr = np.asarray(overlap_arr, dtype=np.float32)
        baseline_dist = np.asarray(baseline_dist_arr, dtype=np.float32)
        # overlap_tensor = torch.stack(overlap_arr, dim=0)
        # baseline_dist_tensor = torch.stack(baseline_dist_arr, dim=0)

        return {'img': img_tensors, 'rel_overlap': torch.from_numpy(overlap_arr), 'rel_baseline': torch.from_numpy(baseline_dist)}


if __name__ == '__main__':

    base_dir = '/mnt/Exp_2/SUN3D_Valid/'
    list_file = '/mnt/Exp_2/SUN3D_Valid/relocal_valid.bin'
    dataset = SUN3DDataset(base_dir, list_file)

    sample = dataset.__getitem__(1)
    print(sample)
