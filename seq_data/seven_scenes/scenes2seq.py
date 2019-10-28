import shutil
import glob, pickle
import os
import numpy as np
import sys
sys.path.append('../../')
sys.path.append('./')
import core_3dv.camera_operator as cam_opt
from frame_seq_data import FrameSeqData
from seq_data.tum_rgbd.tum_seq2ares import export_to_tum_format, export_tum_img_info

def re_organize(seq_path):
    rgb_dir = os.path.join(seq_path, 'rgb')
    if not os.path.exists(rgb_dir):
        os.mkdir(rgb_dir)

    depth_dir = os.path.join(seq_path, 'depth')
    if not os.path.exists(depth_dir):
        os.mkdir(depth_dir)

    poses_dir = os.path.join(seq_path, 'poses')
    if not os.path.exists(poses_dir):
        os.mkdir(poses_dir)

    seqs = sorted(glob.glob(os.path.join(seq_path, '*.pose.txt')))
    seq_names = [seq.split('/')[-1].split('.pose.txt')[0] for seq in seqs]
    for seq_name in seq_names:
        shutil.move(os.path.join(seq_path, '%s.color.png' % seq_name), os.path.join(seq_path, 'rgb', '%s.color.png' % seq_name))
        shutil.move(os.path.join(seq_path, '%s.depth.png' % seq_name), os.path.join(seq_path, 'depth', '%s.depth.png' % seq_name))
        shutil.move(os.path.join(seq_path, '%s.pose.txt' % seq_name), os.path.join(seq_path, 'poses', '%s.pose.txt' % seq_name))
    os.system('rm %s' % os.path.join(seq_path, 'Thumbs.db'))


def scenes2ares(seq_path, seq_name):
    rgb_dir = os.path.join(seq_path, seq_name, 'rgb')
    depth_dir = os.path.join(seq_path, seq_name, 'depth')
    poses_dir = os.path.join(seq_path, seq_name, 'poses')

    frames = sorted(glob.glob(os.path.join(rgb_dir, '*.color.png')))
    frame_names = [seq.split('/')[-1].split('.color.png')[0] for seq in frames]

    default_intrinsic = np.asarray([585, 585, 320, 240, 0, 0], dtype=np.float32)

    frame_seq = FrameSeqData()

    # Read the pose
    for frame_idx, frame_name in enumerate(frame_names):
        pose_file = os.path.join(poses_dir, frame_name + '.pose.txt')
        rgb_file = os.path.join(poses_dir, frame_name + '.color.png')
        depth_file = os.path.join(poses_dir, frame_name + '.depth.png')

        # Read the pose
        pose = np.loadtxt(pose_file).astype(np.float32).reshape(4, 4)
        Tcw = cam_opt.camera_pose_inv(pose[:3, :3], pose[:3, 3])
        timestamp = float(frame_idx)

        frame_seq.append_frame(frame_idx=frame_idx,
                               img_file_name=os.path.join(seq_name, 'rgb', frame_name + '.color.png'),
                               Tcw=Tcw,
                               camera_intrinsic=default_intrinsic,
                               frame_dim=(480, 640),
                               time_stamp=timestamp,
                               depth_file_name=os.path.join(seq_name, 'depth', frame_name + '.depth.png'))

    return frame_seq


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Useage: python scenes2seq.py <7scene_seq_dir>')

    seq_dir = sys.argv[1]
    seq_names = glob.glob(os.path.join(seq_dir, 'seq*'))

    # re-organize sequences
    for seq_name in seq_names:
        if not os.path.isdir(seq_name):
            continue

        if not os.path.isfile(os.path.join(seq_dir, seq_name, 'groundtruth.txt')):
            continue

        re_organize(os.path.join(seq_dir, seq_name))
        seq = scenes2ares(seq_dir, seq_name)
        seq.dump_to_json(os.path.join(seq_dir, seq_name, 'seq.json'))

        export_tum_img_info(seq,
                            os.path.join(seq_dir, seq_name, 'rgb.txt'),
                            os.path.join(seq_dir, seq_name, 'depth.txt'))
        export_to_tum_format(seq,
                             os.path.join(seq_dir, seq_name, 'groundtruth.txt'))

        K = seq.get_K_mat(seq.frames[0])
        np.savetxt(os.path.join(seq_dir, seq_name, 'K.txt'), K)

    # generate train and test frames information (e.g. extrinsic, intrinsic)

    # load train and test split txt
    with open(os.path.join(seq_dir, 'TestSplit.txt')) as f:
        test_seqs_l = f.readlines()
        test_seqs_l = [int(l.split('sequence')[1].strip()) for l in test_seqs_l]

    with open(os.path.join(seq_dir, 'TrainSplit.txt')) as f:
        train_seqs_l = f.readlines()
        train_seqs_l = [int(l.split('sequence')[1].strip()) for l in train_seqs_l]

    # collect all test and train frames from all sequences
    test_frames = []
    for test_seq in test_seqs_l:
        json_path = os.path.join(seq_dir, 'seq-%02d' % test_seq, 'seq.json')
        seq = FrameSeqData(json_path)
        test_frames += seq.frames

    train_frames = []
    for train_seq in train_seqs_l:
        json_path = os.path.join(seq_dir, 'seq-%02d' % test_seq, 'seq.json')
        seq = FrameSeqData(json_path)
        train_frames += seq.frames

    # dump
    with open(os.path.join(seq_dir, 'test_frames.bin')) as f:
        pickle.dump(test_frames, f)

    with open(os.path.join(seq_dir, 'train_frames.bin')) as f:
        pickle.dump(train_frames, f)