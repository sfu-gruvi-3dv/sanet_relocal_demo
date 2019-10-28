import datetime
import os
import sys
import warnings
import argparse
import shutil
import inspect
import numpy as np
import torch.nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
import visualizer.visualizer_2d
from dataset.ours.pytorch_dataset import ImagePairDataset, ba_tracknet_preprocess
from core_dl.train_params import TrainParameters
from core_dl.logger import Logger
import core_dl.module_util as dl_util

from banet_track.ba_tracknet import TrackNet
from banet_track.ba_module import se3_exp, batched_rot2quaternion
import banet_track.ba_tracknet_mirror_a as mirror_a
import banet_track.ba_tracknet_mirror_b as mirror_b

# PySophus Directory
sys.path.extend(['/opt/eigency', '/opt/PySophus'])

""" Configure ----------------------------------------------------------------------------------------------------------
"""
# Set workspace for temp data, checkpoints etc.
workspace_dir = '/mnt/Tango/banet_track_train/'
if not os.path.exists(workspace_dir):
    os.mkdir(workspace_dir)

# Load Checkpoints if needed
checkpoint_path_a = '/mnt/Tango/banet_track_train/logs/Sep23_23-08-07_cs-gruvi-24-cmpt-sfu-ca_r1.25t0.16_itr1_f96_0.01reg_onlylevel2_b4_no_outside_points_quatloss/checkpoints/iter_003691.pth.tar'
checkpoint_a = None if (checkpoint_path_a is None or not os.path.exists(checkpoint_path_a)) \
    else dl_util.load_checkpoints(checkpoint_path_a)

checkpoint_path_b = '/mnt/Tango/banet_track_train/logs/Sep23_23-04-22_cs-gruvi-24-cmpt-sfu-ca_r1.25t0.16_itr1_f96_onlylevel2_b4_no_outside_points_quatloss/checkpoints/iter_003691.pth.tar'
checkpoint_b = None if (checkpoint_path_b is None or not os.path.exists(checkpoint_path_b)) \
    else dl_util.load_checkpoints(checkpoint_path_b)

""" Prepare the dataset ------------------------------------------------------------------------------------------------
"""
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='ba_tracknet', help='name of the dataset')
parser.add_argument('--base_dir', type=str, default='/mnt/Tango/datasets/sun3d_demon', help='path to the data directory')
parser.add_argument('--baseline_threshold', type=float, default=2.0, help='threshold of acceptable baseline')
parser.add_argument('--rotation_threshold', type=float, default=30.0, help='threshold of rotation angle, in degree')
parser.add_argument('--img_extend', type=str, default='.jpg', help='the extension of image files')
parser.add_argument('--phase', type=str, default='train', help='usage of the dataset, train/test/val')
parser.add_argument('--dataset_size', type=int, default=-1, help='size of the dataset, -1 means unknown')
parser.add_argument('--batch_size', type=int, default=1, help='size of 1 batch')
parser.add_argument('--down_level', type=int, default=2, help='down sample rate = 2**down_level')

opt = parser.parse_args()
transforms = transforms.Compose([transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
train_set = ImagePairDataset(opt, transforms)

# Define the dataset transformer
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=1,
                                           shuffle=True,
                                           num_workers=0)

""" Initialize Network -------------------------------------------------------------------------------------------------
"""
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.set_device(0)
net_a = mirror_a.TrackNet(backbone_net='drn_c_26', input_dim=(3, 240, 320), num_pyramid_features=96)
net_a.set_eval_batch_size(1)
if checkpoint_a is not None and 'net_instance' in checkpoint_a.keys():
    net_a.load_state_dict(checkpoint_a['net_instance'])
    print('[Init. Network] Load Net States from checkpoint: ' + checkpoint_path_a)

net_b = mirror_b.TrackNet(backbone_net='drn_c_26', input_dim=(3, 240, 320), num_pyramid_features=96)
net_b.set_eval_batch_size(1)
if checkpoint_b is not None and 'net_instance' in checkpoint_b.keys():
    net_b.load_state_dict(checkpoint_b['net_instance'])
    print('[Init. Network] Load Net States from checkpoint: ' + checkpoint_path_b)

""" Testing  -----------------------------------------------------------------------------------------------------------
# """
for batch_idx, frame_dict in enumerate(train_loader):
    torch.set_default_tensor_type('torch.FloatTensor')
    net_a.eval()
    net_b.eval()
    I_a, d_a, sel_a_indices, K, I_b, q_gt, t_gt, se3_gt = ba_tracknet_preprocess(frame_dict, 3, 1500)
    img_list = []
    N = I_a.shape[0]
    for i in range(N):
        img_list += [{'img': I_a[i, :, :, :].cpu().numpy().transpose(1, 2, 0), 'title': 'frame0'},
                     {'img': I_b[i, :, :, :].cpu().numpy().transpose(1, 2, 0), 'title': 'frame1'}]
    visualizer.visualizer_2d.show_multiple_img(img_list, num_cols=2)

    keyboard = input('s: show feature distance.  n: jump to next image.')
    if keyboard == 's':
        x = int(input('x:'))
        y = int(input('y:'))

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        I_a = I_a.cuda()
        d_a = d_a.cuda()
        sel_a_indices = sel_a_indices.cuda()
        K = K.cuda()
        I_b = I_b.cuda()
        q_gt = q_gt.cuda()
        t_gt = t_gt.cuda()
        se3_gt = se3_gt.cuda()

        net_a.verify_features(I_a, d_a, K, I_b, se3_gt, x, y, title="itr1_large_noise_0.01reg")
        net_b.verify_features(I_a, d_a, K, I_b, se3_gt, x, y, title="itr1_large_noise")