import numpy as np
import os
import glob
import sys
from tqdm import tqdm
from core_3dv.camera_operator import camera_pose_inv
from frame_seq_data import FrameSeqData


def read_sun3d_seq(sun3d_base_dir, seq_name):
    """
    Read the SUN3D sequence to the frames collection
    :param sun3d_seq_dir: input sun3d sequence directory
    :return: uniform frames collection.
    """
    frames = FrameSeqData()
    abs_seq_dir = os.path.join(sun3d_base_dir, seq_name)

    # Read intrinsic mat
    intrinsic_file_path = os.path.join(abs_seq_dir, 'intrinsics.txt')
    if not os.path.exists(intrinsic_file_path):
        raise Exception("DIR: %s ----" % abs_seq_dir)
    K = np.loadtxt(intrinsic_file_path, dtype=np.float32).reshape((3, 3))
    K_param = np.asarray([K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0.0, 0.0], dtype=np.float32)
    default_img_dim = (480, 640)

    # Read extrinsic poses
    ext_pose_file_path = sorted(glob.glob(os.path.join(abs_seq_dir, 'extrinsics', '*.txt')))[-1]
    ext_poses = np.loadtxt(ext_pose_file_path)
    n_frames = int(ext_poses.shape[0] / 3)
    ext_poses = ext_poses.reshape((n_frames, 3, 4))

    # Synchronize the image and depth with timestamp
    depth_list = sorted(glob.glob(os.path.join(abs_seq_dir, 'depth', '*.png')))
    depth_timestamps = []
    for depth_path in depth_list:
        depth_name = depth_path.split('/')[-1].strip()
        depth_tokens = depth_name.split('-')[1].split('.')[0]
        depth_timestamps.append(int(depth_tokens))
    depth_timestamps = np.asarray(depth_timestamps)

    img_list = sorted(glob.glob(os.path.join(abs_seq_dir, 'image', '*.jpg')))
    assert len(img_list) == n_frames
    for frame_idx, img_path in enumerate(img_list):
        img_name = img_path.split('/')[-1].strip()
        img_tokens = img_name.split('-')
        frame_timestamp = int(img_tokens[1].split('.')[0])

        # Find the closest depth frame
        depth_frame_idx = np.argmin(np.abs(depth_timestamps - frame_timestamp))
        depth_frame_path = depth_list[depth_frame_idx].split('/')[-1].strip()

        Twc = ext_poses[frame_idx]
        frames.append_frame(frame_idx=frame_idx,
                            img_file_name=os.path.join(seq_name, 'image', img_name),
                            Tcw=camera_pose_inv(Twc[:3, :3], Twc[:3, 3]),
                            camera_intrinsic=K_param,
                            frame_dim=default_img_dim,
                            time_stamp=float(frame_timestamp),
                            depth_file_name=os.path.join(seq_name, 'depth', depth_frame_path))
    return frames

def sun3d_seq2ares(sun3d_base_dir, seq_name, out_ares_json_path):
    """
    Convert sun3d sequence to ares
    :param sun3d_base_dir: input sun3d sequence directory
    :param out_ares_json_path: output ares format directory
    """
    frames = read_sun3d_seq(sun3d_base_dir, seq_name)
    frames.dump_to_json(out_ares_json_path)


if __name__ == '__main__':

    base_dir = sys.argv[0] if len(sys.argv) > 1 else '/home/ziqianb/Documents/data/SUN3D/'

    list_file = os.path.join(base_dir, 'SUN3Dv1.txt')

    seq_lists = []
    with open(list_file, 'r') as list_f:
        for seq_name in list_f:
            seq_name = seq_name.strip()
            seq_lists.append(seq_name)

    for seq_name in tqdm(seq_lists):
        if os.path.exists(os.path.join(base_dir, seq_name)):
            print(seq_name)
            output_path = os.path.join(base_dir, seq_name, 'seq.json')
            if not os.path.exists(output_path):
                sun3d_seq2ares(base_dir, seq_name=seq_name, out_ares_json_path=output_path)