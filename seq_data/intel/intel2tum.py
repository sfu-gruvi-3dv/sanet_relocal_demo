import shutil
import glob
import os
import numpy as np
import core_3dv.camera_operator as cam_opt
from frame_seq_data import FrameSeqData

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

    seqs = sorted(glob.glob(os.path.join(seq_path, '*.txt')))
    seq_names = [seq.split('/')[-1].split('.txt')[0] for seq in seqs]
    for seq_name in seq_names:
        shutil.move(os.path.join(seq_path, '%s.jpg' % seq_name), os.path.join(seq_path, 'rgb', '%s.jpg' % seq_name))
        shutil.move(os.path.join(seq_path, '%s.png' % seq_name), os.path.join(seq_path, 'depth', '%s.png' % seq_name))
        shutil.move(os.path.join(seq_path, '%s.txt' % seq_name), os.path.join(seq_path, 'poses', '%s.txt' % seq_name))
    os.system('rm %s' % os.path.join(seq_path, 'Thumbs.db'))


def scenes2ares(seq_path, seq_name):
    rgb_dir = os.path.join(seq_path, seq_name, 'rgb')
    depth_dir = os.path.join(seq_path, seq_name, 'depth')
    poses_dir = os.path.join(seq_path, seq_name, 'poses')

    frames = sorted(glob.glob(os.path.join(rgb_dir, '*.jpg')))
    frame_names = [seq.split('/')[-1].split('.jpg')[0] for seq in frames]

    default_intrinsic = np.asarray([525, 525, 320, 240, 0, 0], dtype=np.float32)

    frame_seq = FrameSeqData()

    # Read the pose
    for frame_idx, frame_name in enumerate(frame_names):
        pose_file = os.path.join(poses_dir, frame_name + '.txt')
        rgb_file = os.path.join(poses_dir, frame_name + '.jpg')
        depth_file = os.path.join(poses_dir, frame_name + '.png')

        # Read the pose
        pose = np.loadtxt(pose_file).astype(np.float32).reshape(4, 4)
        Tcw = cam_opt.camera_pose_inv(pose[:3, :3], pose[:3, 3])
        timestamp = float(frame_idx)

        frame_seq.append_frame(frame_idx=frame_idx,
                               img_file_name=os.path.join(seq_name, 'rgb', frame_name + '.jpg'),
                               Tcw=Tcw,
                               camera_intrinsic=default_intrinsic,
                               frame_dim=(480, 640),
                               time_stamp=timestamp,
                               depth_file_name=os.path.join(seq_name, 'depth', frame_name + '.png'))

    return frame_seq


if __name__ == '__main__':
    seq_dir = '/home/ziqianb/Desktop/intel/'
    seq_name = 'apartment'
    # re_organize(os.path.join(seq_dir, seq_name))
    seq = scenes2ares(seq_dir, seq_name)
    seq.frames = seq.frames[18000:19600]
    # Reset the frame index
    for frame_idx, frame in enumerate(seq.frames):
        frame['id'] = frame_idx
    seq.dump_to_json(os.path.join(seq_dir, seq_name, 'seq_2.json'))

    from seq_data.tum_rgbd.tum_seq2ares import export_to_tum_format, export_tum_img_info
    export_tum_img_info(seq,
                        os.path.join(seq_dir, seq_name, 'rgb.txt'),
                        os.path.join(seq_dir, seq_name, 'depth.txt'))
    export_to_tum_format(seq,
                         os.path.join(seq_dir, seq_name, 'groundtruth.txt'))

    K = seq.get_K_mat(seq.frames[0])
    np.savetxt(os.path.join(seq_dir, seq_name, 'K.txt'), K)

    # print(seq.frames[:2])