import shutil
import glob
import os
import numpy as np
import re
from tqdm import tqdm
import core_3dv.camera_operator as cam_opt
from frame_seq_data import FrameSeqData

def re_organize(base_dir, scene_name):
    scene_dir = os.path.join(base_dir, scene_name)
    frames = sorted(glob.glob(os.path.join(scene_dir, 'data', '*.pose.txt')))
    frame_names = [frame.split('/')[-1].split('.pose.txt')[0] for frame in frames]

    seq_list = []
    with open(os.path.join(scene_dir, 'split.txt'), 'r') as f:
        i = 0
        for line in f:
            # Make dir for seq-i
            seq_dir = os.path.join(scene_dir, 'seq-%02d' % i)
            if not os.path.exists(seq_dir):
                os.mkdir(seq_dir)
            seq_list.append(os.path.join(scene_name, 'seq-%02d' % i))

            # Get frame indices for seq-i
            start_idx = int(re.findall(r'\[start=([0-9]*) ', line)[0])
            end_idx = int(re.findall(r'end=([0-9]*)\]', line)[0]) + 1

            # Move rgb, depth, pose to seq-i dir
            rgb_dir = os.path.join(seq_dir, 'rgb')
            if not os.path.exists(rgb_dir):
                os.mkdir(rgb_dir)
            depth_dir = os.path.join(seq_dir, 'depth')
            if not os.path.exists(depth_dir):
                os.mkdir(depth_dir)
            poses_dir = os.path.join(seq_dir, 'poses')
            if not os.path.exists(poses_dir):
                os.mkdir(poses_dir)
            for frame_name in frame_names[start_idx : end_idx]:
                # print(os.path.join(scene_dir, 'data', '%s.color.jpg' % frame_name), os.path.join(rgb_dir, '%s.color.jpg' % frame_name))
                # print(os.path.join(scene_dir, 'data', '%s.depth.png' % frame_name), os.path.join(depth_dir, '%s.depth.png' % frame_name))
                # print(os.path.join(scene_dir, 'data', '%s.pose.txt' % frame_name), os.path.join(poses_dir, '%s.pose.txt' % frame_name))
                if os.path.exists(os.path.join(scene_dir, 'data', '%s.color.jpg' % frame_name)):
                    shutil.move(os.path.join(scene_dir, 'data', '%s.color.jpg' % frame_name), os.path.join(rgb_dir, '%s.color.jpg' % frame_name))
                    shutil.move(os.path.join(scene_dir, 'data', '%s.depth.png' % frame_name), os.path.join(depth_dir, '%s.depth.png' % frame_name))
                    shutil.move(os.path.join(scene_dir, 'data', '%s.pose.txt' % frame_name), os.path.join(poses_dir, '%s.pose.txt' % frame_name))

            i += 1
    return seq_list


def scenes2ares(base_dir, seq_name):
    rgb_dir = os.path.join(base_dir, seq_name, 'rgb')
    depth_dir = os.path.join(base_dir, seq_name, 'depth')
    poses_dir = os.path.join(base_dir, seq_name, 'poses')

    frames = sorted(glob.glob(os.path.join(rgb_dir, '*.color.jpg')))
    frame_names = [seq.split('/')[-1].split('.color.jpg')[0] for seq in frames]

    default_intrinsic = np.asarray([572, 572, 320, 240, 0, 0], dtype=np.float32)
    default_rgb_intrinsic = np.asarray([1158.3, 1153.53, 649, 483.5, 0, 0], dtype=np.float32)

    frame_seq = FrameSeqData()

    # Read the pose
    frame_idx = 0
    for i, frame_name in enumerate(frame_names):
        pose_file = os.path.join(poses_dir, frame_name + '.pose.txt')

        INF_flag = False
        with open(pose_file, 'r') as f:
            lines = f.readlines()
            if 'INF' in lines[0]:
                INF_flag = True
        if INF_flag:
            continue

        # Read the pose
        pose = np.loadtxt(pose_file).astype(np.float32).reshape(4, 4)
        Tcw = cam_opt.camera_pose_inv(pose[:3, :3], pose[:3, 3])
        timestamp = float(i)

        frame_seq.append_frame(frame_idx=frame_idx,
                               img_file_name=os.path.join(seq_name, 'rgb', frame_name + '.color.jpg'),
                               Tcw=Tcw,
                               camera_intrinsic=default_intrinsic,
                               frame_dim=(480, 640),
                               time_stamp=timestamp,
                               depth_file_name=os.path.join(seq_name, 'depth', frame_name + '.depth.png'),
                               rgb_intrinsic=default_rgb_intrinsic)
        frame_idx += 1

    return frame_seq


if __name__ == '__main__':
    base_dir = '/home/luwei/mnt/Exp_1/12scenes'
    seq_list = []
    building_list = os.listdir(base_dir)
    for building_name in building_list:
        building_dir = os.path.join(base_dir, building_name)
        if not os.path.isdir(building_dir):
            continue
        scene_list = os.listdir(building_dir)
        for scene_name in scene_list:
            scene_dir = os.path.join(building_dir, scene_name)
            if not os.path.isdir(scene_dir):
                continue
            seq_list += re_organize(base_dir, os.path.join(building_name, scene_name))

    for seq_name in tqdm(seq_list):
        seq = scenes2ares(base_dir, seq_name)
        seq.dump_to_json(os.path.join(base_dir, seq_name, 'seq.json'))

    # seq_name = 'chess/seq-04'
    # re_organize(os.path.join(seq_dir, seq_name))

    #
    # from seq_data.tum_rgbd.tum_seq2ares import export_to_tum_format, export_tum_img_info
    # export_tum_img_info(seq,
    #                     os.path.join(seq_dir, seq_name, 'rgb.txt'),
    #                     os.path.join(seq_dir, seq_name, 'depth.txt'))
    # export_to_tum_format(seq,
    #                      os.path.join(seq_dir, seq_name, 'groundtruth.txt'))
    #
    # K = seq.get_K_mat(seq.frames[0])
    # np.savetxt(os.path.join(seq_dir, seq_name, 'K.txt'), K)

    # print(seq.frames[:2])