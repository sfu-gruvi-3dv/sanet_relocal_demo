from relocal.vlad_module import TripletRankingLoss
import torch
import pickle
import tensorboardX

print(dir(tensorboardX))

# with open('/mnt/Exp_2/SUN3D_Valid/relocal_valid.bin', 'rb') as f:
#     t_list = pickle.load(f, encoding='latin1')
#     t_list = t_list[:10]
#
# criterion = TripletRankingLoss(margin=0.1, triplet_config={'overlap_thres': 0.80,
#                                                            'baseline_thres': 0.08,
#                                                            'anchor_ref_idx': 3})
