import os
import sys
import pickle
import random
import copy
from tqdm import tqdm

from frame_seq_data import FrameSeqData


if __name__ == '__main__':
    
    out_bin_file = '/mnt/Exp_2/SUN3D_Valid/reloc_pairs_Scene2QueryOverlap0.5to0.55_Rot0to10_valid.bin'
    shuffle_list = False
    
    with open('/mnt/Exp_2/SUN3D_Valid/reloc_pairs_Overlap0.5to0.55_Rot0to10_valid.bin', 'rb') as f:
        data_list = pickle.load(f)

    reversed_data_list = []
    for sample_dict in tqdm(data_list):
        sub_frames = sample_dict['sub_frames']
        train_anchor_frames = sample_dict['train_anchor_frames']
        test_anchor_frames = sample_dict['test_anchor_frames']
        
        assert len(test_anchor_frames) == 0
        assert len(sub_frames) == 1
        assert len(train_anchor_frames) == 1
        
        res_sub_frames = FrameSeqData()
        res_sub_frames.frames.append(copy.deepcopy(train_anchor_frames[0]))
        res_train_anchor_frames = [copy.deepcopy(sub_frames.frames[0])]
        
        reversed_data_list.append({'sub_frames': res_sub_frames, 'train_anchor_frames': res_train_anchor_frames, 'test_anchor_frames': []})
    
    with open(out_bin_file, 'wb') as out_f:
        if shuffle_list:
            random.shuffle(reversed_data_list)
        pickle.dump(reversed_data_list, out_f)
        print('Total:', len(reversed_data_list))
        print('Done, saved to %s' % out_bin_file)
