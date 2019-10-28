from frame_seq_data import FrameSeqData
from core_io.lmdb_writer import LMDBWriter
import os
import cv2
from tqdm import tqdm
from seq_data.sun3d.read_util import read_sun3d_depth
import numpy as np
import matplotlib.pyplot as plt


def read_raw_depth_uint16(filename):
    depth_pil = cv2.imread(filename, cv2.IMREAD_ANYDEPTH).astype(np.uint16)
    depth_shifted = (depth_pil >> 3) | (depth_pil << 13)
    return depth_shifted

""" Configuration
"""

dataset_dir = '/local-scratch/SUN3D'

seq_name_path = 'SUN3Dv1_train.txt'

lmdb_output_dir = '/local-scratch3/SUN3D'

""" LMDB Seq Model
"""
from core_io.lmdb_reader import LMDBModel
class LMDBSeqModel(LMDBModel):

    def __init__(self, lmdb_path):
        super(LMDBSeqModel, self).__init__(lmdb_path)

    def read_depth(self, depth_key, min_depth_thres=1e-5):
        depth_str = np.fromstring(self.read_by_key(depth_key), dtype=np.uint8)
        depth = np.asarray(cv2.imdecode(depth_str, cv2.IMREAD_ANYDEPTH)).reshape((240, 320))
        # print(depth.dtype)
        depth = depth.astype(np.float32)
        depth = (depth / 1000)
        # depth[depth < min_depth_thres] = min_depth_thres
        return depth

    def read_img(self, img_key):
        img_str = np.fromstring(self.read_by_key(img_key), dtype=np.uint8)
        img = np.asarray(cv2.imdecode(img_str, cv2.IMREAD_ANYCOLOR)).reshape((240, 320))
        return img

""" Scripts
"""
if __name__ == '__main__':

    # Read list
    list_file = os.path.join(dataset_dir, seq_name_path)
    seq_name_list = []
    with open(list_file, 'r') as list_f:
        for seq_name in list_f:
            seq_name = seq_name.strip()
            seq_name_list.append(seq_name)

    for seq_name in tqdm(seq_name_list[-1:], desc='generating lmdbs for sequences'):
        seq_file_path = os.path.join(dataset_dir, seq_name, 'seq.json')
        if not os.path.exists(seq_file_path):
            continue
        seq = FrameSeqData(seq_file_path)

        seq_lmdb = LMDBSeqModel(os.path.join(dataset_dir, seq_name, 'rgbd.lmdb'))
        for frame_idx in range(0, 80, 20):
            frame = seq.frames[frame_idx]
            img_path = os.path.join(dataset_dir, seq.get_image_name(frame))
            img2 = cv2.imread(img_path)
            depth_path = os.path.join(dataset_dir, seq.get_depth_name(frame))
            depth = read_sun3d_depth(depth_path)
            depth = cv2.resize(depth, (320, 240), interpolation=cv2.INTER_NEAREST)

            img_key = seq.get_image_name(frame)
            depth_key = seq.get_depth_name(frame)

            img = seq_lmdb.read_img(img_key)
            depth2 = seq_lmdb.read_depth(depth_key)

            plt.imshow(depth, cmap='jet')
            plt.show()
            plt.imshow(depth2, cmap='jet')
            plt.show()

        seq_lmdb.close_session()

    for seq_name in seq_name_list:
        seq_file_path = os.path.join(dataset_dir, seq_name, 'seq.json')
        if not os.path.exists(seq_file_path):
            continue
        seq = FrameSeqData(seq_file_path)

        if not os.path.exists(os.path.join(lmdb_output_dir, seq_name)):
            os.makedirs(os.path.join(lmdb_output_dir, seq_name))

        lmdb = LMDBWriter(os.path.join(lmdb_output_dir, seq_name, 'rgbd.lmdb'))
        for frame in tqdm(seq.frames, desc='processing: ' + seq_name):
            # img_path = os.path.join(dataset_dir, seq.get_image_name(frame))
            # img = cv2.imread(img_path)
            # img = cv2.resize(img, (320, 240))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img_str = cv2.imencode('.jpg', img)[1].tostring()
            # lmdb.write_str(seq.get_image_name(frame), img_str)

            depth_path = os.path.join(dataset_dir, seq.get_depth_name(frame))
            depth = read_raw_depth_uint16(depth_path)
            depth = cv2.resize(depth, (320, 240), interpolation=cv2.INTER_NEAREST)
            depth_str = cv2.imencode('.png', depth)[1].tostring()
            lmdb.write_str(seq.get_depth_name(frame), depth_str)

        lmdb.close_session()
