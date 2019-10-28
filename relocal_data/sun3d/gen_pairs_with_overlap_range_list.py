import sys
import os
from tqdm import tqdm
from relocal_data.sun3d.random_sel_module import sel_pairs_with_overlap_range_sun3d
from frame_seq_data import FrameSeqData
import torch as T
import pickle
import random
from relocal_data.sun3d.gen_lmdb_cache import LMDBSeqModel


''' Configuration ------------------------------------------------------------------------------------------------------
'''
trans_thres = 2.0

rotation_threshold = (20.0, 999.0)

overlap_threshold = 0.2#(0.5, 0.55)

scene_dist_threshold = (0.0, 1.0)

max_sub_seq_per_scene = 0.1

frames_per_sub_seq = 2

skip_frames = 1

shuffle_list = False

train_anchor_num = 10

test_anchor_num = 0

''' Utilities ----------------------------------------------------------------------------------------------------------
'''
def check_file_exist(basedir, frames:FrameSeqData):
    exist_flag = True
    for frame in frames.frames:
        img_name = frame['file_name']
        depth_name = frame['depth_file_name']
        if not os.path.exists(os.path.join(basedir, img_name)):
            exist_flag = False
            break
        if not os.path.exists(os.path.join(basedir, depth_name)):
            exist_flag = False
            break
    return exist_flag


''' Script -------------------------------------------------------------------------------------------------------------
'''
if __name__ == '__main__':

    base_dir = sys.argv[1] if len(sys.argv) > 1 else '/mnt/Exp_2/SUN3D'
    out_bin_file = sys.argv[2] if len(sys.argv) > 2 else '/mnt/Exp_2/SUN3D/reloc_subseq2_Scene2SceneOverlap0.2_Dist2.0_train.bin'
    lmdb_dir = sys.argv[3] if len(sys.argv) > 3 else '/mnt/Exp_3/SUN3D'

    # Read list
    list_file = os.path.join(base_dir, 'SUN3Dv1_train.txt')
    seq_lists = []
    with open(list_file, 'r') as list_f:
        for seq_name in list_f:
            seq_name = seq_name.strip()
            seq_lists.append(seq_name)

    total_sub_seq_list = []
    for seq_name in tqdm(seq_lists):
        in_frame_path = os.path.join(base_dir, seq_name, 'seq.json')
        if os.path.exists(in_frame_path):
            frames = FrameSeqData(in_frame_path)
            frame_lmdb = LMDBSeqModel(os.path.join(lmdb_dir, seq_name, 'rgbd.lmdb'))
            if check_file_exist(base_dir, frames) is False:
                print('None Exist on %s', seq_name)
            else:
                sub_frames_list = sel_pairs_with_overlap_range_sun3d(frames,
								                                     frame_lmdb,
                                                                     trans_thres=trans_thres,
                                                                     rot_thres=rotation_threshold,
                                                                     dataset_base_dir=base_dir,
                                                                     overlap_thres=overlap_threshold,
                                                                     scene_dist_thres=scene_dist_threshold,
                                                                     max_subseq_num=max_sub_seq_per_scene,
                                                                     frames_per_subseq_num=frames_per_sub_seq,
                                                                     frames_range=(0.02, 0.9),
                                                                     interval_skip_frames=skip_frames,
                                                                     train_anchor_num=train_anchor_num,
                                                                     test_anchor_num=test_anchor_num)
                total_sub_seq_list += sub_frames_list
                # if len(sub_frames_list) > 0:
                #     print(sub_frames_list[0]['sub_frames'].frames)
        # break


    with open(out_bin_file, 'wb') as out_f:
        if shuffle_list:
            random.shuffle(total_sub_seq_list)
        pickle.dump(total_sub_seq_list, out_f)
        print('Total:', len(total_sub_seq_list))
        print('Done, saved to %s' % out_bin_file)



