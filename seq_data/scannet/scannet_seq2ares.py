import os
import numpy as np
import sys
import glob
from tqdm import tqdm
from core_3dv.camera_operator import camera_pose_inv
from frame_seq_data import FrameSeqData


def read_scannet_seq(scannet_base_dir, seq_name):
    """
    Read SCANNET sequences to the frames collection
    :param input_seq_dir: directory single SCANNET sequence.
    :return: uniform frames collection.
    """
    frames = FrameSeqData()

    abs_seq_dir = os.path.join(scannet_base_dir, seq_name)

    # Read camera intrinsic info.
    intrinsic_txt = os.path.join(abs_seq_dir, 'info.txt')
    if not os.path.exists(intrinsic_txt):
        raise Exception("No camera intrinsic mat.")
    with open(intrinsic_txt, "r") as intrinsic_file:
        for line in intrinsic_file:
            tokens = line.split(' = ')
            if tokens[0].startswith("m_depthWidth"):
                frame_w = int(tokens[1].strip())
            elif tokens[0].startswith("m_depthHeight"):
                frame_h = int(tokens[1].strip())
            elif tokens[0].startswith("m_depthShift"):
                shift_factor = float(tokens[1].strip())
                if shift_factor != 1000:
                    raise Exception("Depth shift error")
            elif tokens[0].startswith("m_calibrationDepthIntrinsic"):
                k_tokens = tokens[1].split(' ')[:16]
                K = np.asarray(k_tokens, dtype=np.float32).reshape(4, 4)
                K_param = np.asarray([K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0.0, 0.0], dtype=np.float32)

    samples_txt = os.path.join(abs_seq_dir, 'samples.txt')
    if not os.path.exists(samples_txt):
        raise Exception("No seq samples info.")
    with open(samples_txt, "r") as sample_file:
        for line in sample_file:
            tokens = line.split(' ')
            tokens[-1] = tokens[-1].strip()
            frame_idx = int(tokens[0])
            frame_name = tokens[1].split('/')[1].split('.')[0]
            img_name = os.path.join(seq_name, 'rgb', frame_name + '.color.jpg')
            depth_name = os.path.join(seq_name, 'depth', frame_name + '.depth.png')
            Twc = np.asarray(tokens[3:19], dtype=np.float32).reshape((4, 4))
            Tcw = np.linalg.inv(Twc)[:3, :]
            frames.append_frame(frame_idx, img_name, Tcw, K_param, (frame_h, frame_w), depth_file_name=depth_name)

    return frames


def scannet_seq2ares(scannet_base_dir, seq_name, output_ares_json_path):
    frames = read_scannet_seq(scannet_base_dir, seq_name)
    frames.dump_to_json(output_ares_json_path)


if __name__ == '__main__':
    scannet_base_dir = sys.argv[1] if len(sys.argv) > 1 else '/mnt/Exp_3/scannet'
    if not os.path.exists(scannet_base_dir):
        raise Exception("No Scannet dir exist.")

    scene_dir_list = glob.glob(os.path.join(scannet_base_dir, 'scene*'))
    if len(scene_dir_list) == 0:
        raise Exception("No seq found.")

    for scene_dir in tqdm(scene_dir_list):
        scene_name = scene_dir.split('/')[-1].strip()
        if not os.path.isdir(scene_dir):
            continue
        output_json_file = os.path.join(scene_dir, 'seq.json')
        scannet_seq2ares(scannet_base_dir, scene_name, output_json_file)

