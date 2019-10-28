import numpy as np
import os
import glob
from core_3dv.camera_operator import camera_pose_inv
from frame_seq_data import FrameSeqData
import core_math.transfom as trans
import core_3dv.camera_operator as cam_opt

def export_tum_img_info(frames: FrameSeqData, rgb_output_path, depth_output_path, comment=None):

    with open(rgb_output_path, 'w') as out_f:
        if comment is not None:
            out_f.write('# ' + comment + '\n')

        for frame in frames.frames:
            timestamp = str(frame['timestamp'])
            if timestamp == 'None':
                continue
            tokens = frame['file_name'].split('/')
            img_dir_name = tokens[-2]
            img_name = tokens[-1]
            # depth_name = frame['depth_file_name']
            out_f.write(timestamp + ' ')
            out_f.write(os.path.join(img_dir_name, img_name) + '\n')

    with open(depth_output_path, 'w') as out_f:
        if comment is not None:
            out_f.write('# ' + comment + '\n')

        for frame in frames.frames:
            timestamp = str(frame['timestamp'])
            if timestamp == 'None':
                continue
            tokens = frame['depth_file_name'].split('/')
            depth_dir_name = tokens[-2]
            depth_name = tokens[-1]
            out_f.write(timestamp + ' ')
            out_f.write(os.path.join(depth_dir_name, depth_name) + '\n')


def export_to_tum_format(frames: FrameSeqData, output_path, comment=None, write_img_info=False):
    """
    Export the frame collection into tum format
    :param frames: frame collection, instance of FrameSeqData
    :param output_path: file with frames, in tum format
    :param comment: comment string put into the header line after '#'
    """
    with open(output_path, 'w') as out_f:
        if comment is not None:
            out_f.write('# ' + comment + '\n')

        for frame in frames.frames:
            Tcw = frame['extrinsic_Tcw']
            timestamp = str(frame['timestamp'])
            if timestamp == 'None':
                continue
            img_name = frame['file_name']
            depth_name = frame['depth_file_name']
            Twc = cam_opt.camera_pose_inv(Tcw[:3, :3], Tcw[:3, 3])

            t = Twc[:3, 3]
            q = trans.quaternion_from_matrix(Twc)

            if write_img_info:
                out_f.write(timestamp + ' ')
                out_f.write(img_name + ' ')
                out_f.write(timestamp + ' ')
                out_f.write(depth_name + ' ')

            out_f.write(timestamp + ' ')
            for t_idx in range(0, 3):
                out_f.write(str(t[t_idx]) + ' ')
            for q_idx in range(1, 4):
                out_f.write(str(q[q_idx]) + ' ')
            out_f.write(str(q[0]))                   # qw in the end
            out_f.write('\n')


def read_tum_seq(tum_rgbd_base_dir, seq_name):
    """
    Read the SUN3D sequence to the frames collection
    :param sun3d_seq_dir: input sun3d sequence directory
    :return: uniform frames collection.
    """
    frames = FrameSeqData()
    abs_seq_dir = os.path.join(tum_rgbd_base_dir, seq_name)

    # Read intrinsic mat
    fx = 525.0                      # focal length x
    fy = 525.0                      # focal length y
    cx = 319.5                      # optical center x
    cy = 239.5                      # optical center y
    K_param = np.asarray([fx, fy, cx, cy, 0.0, 0.0], dtype=np.float32)
    default_img_dim = (480, 640)

    if os.path.exists(os.path.join(abs_seq_dir, 'rdpose_associate.txt')):
        gt_file = 'rdpose_associate.txt'
    else:
        gt_file = 'rd_associate.txt'

    frame_idx = 0
    with open(os.path.join(abs_seq_dir, gt_file), 'r') as f:
        for line in f:
            # Load frame data
            if gt_file.startswith('rdpose_associate'):
                timestamp, img_file_name, _, depth_file_name, _, tx, ty, tz, qx, qy, qz, qw = line.strip().split(' ')
                tx = float(tx)
                ty = float(ty)
                tz = float(tz)
                qx = float(qx)
                qy = float(qy)
                qz = float(qz)
                qw = float(qw)
                R_mat = trans.quaternion_matrix([qw, qx, qy, qz]).astype(np.float32)
                t = np.array([tx, ty, tz]).astype(np.float32)
                Twc_mat = R_mat
                Twc_mat[:3, 3] = t
                Tcw = np.linalg.inv(Twc_mat)[:3, :]
            else:
                timestamp, img_file_name, _, depth_file_name = line.strip().split(' ')
                Tcw = np.eye(4)[:3, :]

            frames.append_frame(frame_idx=frame_idx,
                                img_file_name=os.path.join(seq_name, img_file_name),
                                Tcw=Tcw[:3, :],
                                camera_intrinsic=K_param,
                                frame_dim=default_img_dim,
                                time_stamp=float(timestamp),
                                depth_file_name=os.path.join(seq_name, depth_file_name))

            frame_idx += 1

    return frames


def tum_rgbd_seq2ares(tum_rgbd_base_dir, seq_name, out_ares_json_path):
    """
    Convert TUM-RGBD sequence to ares
    :param sun3d_base_dir: input sun3d sequence directory
    :param out_ares_json_path: output ares format directory
    """
    frames = read_tum_seq(tum_rgbd_base_dir, seq_name)
    frames.dump_to_json(out_ares_json_path)

if __name__ == '__main__':
    base_dir = '/home/ziqianb/Desktop/tgz/'
    seq_name = 'rgbd_dataset_freiburg1_room'
    import glob
    # dirs = glob.glob(os.path.join(base_dir, '*'))
    # for dir in dirs:
    #     if os.path.isdir(dir):
    #         seq_name = dir.split('/')[-1]
    print(seq_name)
    tum_rgbd_seq2ares(base_dir,
                      seq_name,
                      out_ares_json_path=os.path.join(base_dir, seq_name, 'seq.json'))