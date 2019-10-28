import datetime
import os
import sys
import warnings
import pickle
import argparse
import shutil
import inspect
import numpy as np
import scipy.ndimage.filters as fi
import torch.nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
import visualizer.visualizer_2d
from dataset.preprocess import ba_tracknet_preprocess
from dataset.cta73_set.miniscannet_set import MiniScanNetSet, split_train_valid
from core_dl.train_params import TrainParameters
from core_dl.logger import Logger
import core_dl.module_util as dl_util

from banet_track.ba_tracknet import TrackNet
from banet_track.ba_module import se3_exp, batched_rot2quaternion, quaternion_dist, batched_rot2angle, batched_index_select

# PySophus Directory
sys.path.extend(['/opt/eigency', '/opt/PySophus'])

""" Configure ----------------------------------------------------------------------------------------------------------
"""
# Set workspace for temp data, checkpoints etc.
workspace_dir = '/mnt/Tango/banet_track_train/'
if not os.path.exists(workspace_dir):
    os.mkdir(workspace_dir)

# Load Train parameters
train_params = TrainParameters(from_json_file=os.path.join(workspace_dir, 'train_param.json'))

# Load Checkpoints if needed
checkpoint_path = '/mnt/Tango/banet_track_train/logs/Sep28_12-27-29_cs-gruvi-24-cmpt-sfu-ca_scannet_Iinit_drn26_f64_240320_iter3_level3_lr1e-4_batch4/checkpoints/iter_008000.pth.tar'
checkpoint = None if (checkpoint_path is None or not os.path.exists(checkpoint_path)) \
    else dl_util.load_checkpoints(checkpoint_path)

# Setup log dir
log_base_dir = os.path.join(workspace_dir, 'logs')
# log_base_dir = None
log_comment_msg = 'scannet_Iinit_drn26_f64_240320_iter3_level3_lr5e-5_batch4'
if log_base_dir is not None:
    # Setup the logger if dir has provided
    logger = Logger(base_dir=log_base_dir, log_types='tensorboard', comment=log_comment_msg, continue_log=False)
    if train_params.VERBOSE_MODE:
        print('Log dir: %s' % logger.log_base_dir)
else:
    logger = None

# Device for training
dev_id = 1

# Dataset
pair_path = '/mnt/Exp_3/scannet_sel_largebaseline.bin'

img_dir = '/mnt/Exp_3/scannet'

""" Prepare the dataset ------------------------------------------------------------------------------------------------
"""
# Default Transform for DRN26
transforms = transforms.Compose([transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

with open(pair_path, 'rb') as f:
    pairs = pickle.load(f)
train_pairs, valid_pairs = split_train_valid(pairs, train_ratio=0.95)

# Training Set
train_set = MiniScanNetSet(train_pairs, img_dir, transform=transforms, rand_flip=False)
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=train_params.BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=train_params.TORCH_DATALOADER_NUM)

# Validation Set
valid_set = MiniScanNetSet(valid_pairs, img_dir, transform=transforms, rand_flip=False)
valid_loader = torch.utils.data.DataLoader(valid_set,
                                           batch_size=train_params.BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=train_params.TORCH_DATALOADER_NUM)

if train_params.VERBOSE_MODE:
    print("[Dataset Overview] ----------------------------------------------------------------------------------------")
    print("Traing %d items, Valid: %d items" % (len(train_set), len(valid_set)))


""" Initialize Network -------------------------------------------------------------------------------------------------
"""
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.set_device(dev_id)
net = TrackNet(backbone_net='drn_c_26', input_dim=(3, 240, 320), num_pyramid_features=64)
net.set_train_batch_size(train_params.BATCH_SIZE)
if checkpoint is not None and 'net_instance' in checkpoint.keys():
    net.load_state_dict(checkpoint['net_instance'])
    print('[Init. Network] Load Net States from checkpoint: ' + checkpoint_path)


""" Init Optimizer -----------------------------------------------------------------------------------------------------
"""
optimizer = torch.optim.Adam(net.parameters(), lr=train_params.START_LR)
if checkpoint is not None and 'optimizer' in checkpoint.keys():
    optimizer.load_state_dict(checkpoint['optimizer'])

if train_params.VERBOSE_MODE:
    print("[Optimizer Overview] --------------------------------------------------------------------------------------")
    print("[%s] Start learning rate: %f" % (type(optimizer), train_params.START_LR))

""" Configure the Logger -----------------------------------------------------------------------------------------------
"""
if logger is not None:
    logger.add_keys(['Epoch',
                     'Iteration',
                     'Net',
                     'Loss(Train)/rot_loss',
                     'Loss(Train)/trans_loss',
                     'Loss(Train)/reg_loss',
                     'Loss(Train)/total_loss',
                     'Accuracy(Train)/final_rot_error',
                     'Accuracy(Train)/final_trans_error',
                     'Accuracy(Train)/init_rot_error',
                     'Accuracy(Train)/init_trans_error',
                     'Accuracy(Train)/rel_rot_error',
                     'Accuracy(Train)/rel_trans_error',
                     'Accuracy(Train)/final_flow_error',
                     'Accuracy(Valid)/final_rot_error',
                     'Accuracy(Valid)/final_trans_error',
                     'Accuracy(Valid)/init_rot_error',
                     'Accuracy(Valid)/init_trans_error',
                     'Accuracy(Valid)/rel_rot_error',
                     'Accuracy(Valid)/rel_trans_error',
                     'Accuracy(Valid)/final_flow_error',
                     'Histogram(Train)/flow',
                     ])
    # logger.draw_architecture(net, net.input_shape, verbose=False)

    # Save the Network definition
    network_py_path = inspect.getfile(net.__class__)
    shutil.copy(network_py_path, logger.log_base_dir)
    shutil.copy(os.path.realpath(__file__), logger.log_base_dir)

""" Training Preparation -----------------------------------------------------------------------------------------------
"""
# Prepare save model dir
model_checkpoint_dir = os.path.join(logger.log_base_dir if logger is not None else workspace_dir, 'checkpoints')
if not os.path.exists(model_checkpoint_dir):
    os.mkdir(model_checkpoint_dir)

""" Loss Function ------------------------------------------------------------------------------------------------------
"""
def pose_loss(pred_pose_list, q_gt, t_gt, weight):
    N = q_gt.shape[0]
    rot_loss = 0.0
    trans_loss = 0.0
    for level, level_pred_pose in enumerate(pred_pose_list):
        for i, pred_pose in enumerate(level_pred_pose):
            R, t = pred_pose[:, :3, :3], pred_pose[:, :3, 3]
            q = batched_rot2quaternion(R)
            rot_loss += relative_quarternion_loss(q, q_gt) * weight[level * len(level_pred_pose) + i]
            trans_loss += F.mse_loss(t.view(N, 3), t_gt.view(N, 3)) * weight[level * len(level_pred_pose) + i]
    return rot_loss, trans_loss

def relative_quarternion_loss(q1, q2):
    return torch.mean(quaternion_dist(q1, q2))

def relative_rotation_angle_loss(R, R_gt):
    N = R_gt.shape[0]
    angle_gt, axis_gt = batched_rot2angle(R_gt)
    angle, axis = batched_rot2angle(R)
    axis_gt *= angle_gt.view(N, 1)
    axis *= angle.view(N, 1)
    return F.mse_loss(axis, axis_gt)

def feature_reg_loss(gt_f_pairs):
    feature_map_loss = 0.0
    for (f_a, gt_f_wrap_b) in gt_f_pairs:
        e = f_a - gt_f_wrap_b
        feature_map_loss += torch.mean(e * e)
    return feature_map_loss

def feature_region_reg_loss(gt_f_patch_pairs):
    # generate a gaussian kernel
    inp = np.zeros((5, 5), dtype=np.float32)
    inp[2, 2] = 1
    gaussian_kernel = fi.gaussian_filter(inp, 1.0)
    target = torch.cuda.FloatTensor(gaussian_kernel).view(1, 5 * 5, 1)
    target -= target.min()
    target /= target.max()
    target = 1.0 - target
    loss = 0.0
    for (unfold_f_a_sel, unfold_gt_f_wrap_b_sel) in gt_f_patch_pairs:
        N, C, _, M = unfold_f_a_sel.shape
        e = torch.norm(unfold_f_a_sel - unfold_gt_f_wrap_b_sel, p=2, dim=1)                            # (N, 5*5, num_patches)
        # flag0 = torch.isnan(e).any()
        e = e - torch.min(e, dim=1, keepdim=True)[0]
        # flag1 = torch.isnan(e).any()
        e = e / (torch.max(e, dim=1, keepdim=True)[0] + 1e-4)
        # flag2 = torch.isnan(e).any()
        loss += torch.mean((target - e) ** 2)
    return loss


def relative_angle(q1, q2):
    return 180 * torch.acos(2*torch.sum(q1 * q2, dim=-1) ** 2 - 1) / np.pi


def weighted_feature_reg_loss(gt_f_pairs, sel_a_indices):
    # generate a gaussian kernel
    inp = np.zeros((7, 7), dtype=np.float32)
    inp[3, 3] = 1
    gaussian_kernel = fi.gaussian_filter(inp, 2.0)
    kernel = torch.cuda.FloatTensor(gaussian_kernel).view(1, 1, 7, 7)
    kernel -= kernel.min()
    kernel /= kernel.max()
    kernel -= 0.75
    M = sel_a_indices.shape[2]
    feature_map_loss = 0.0
    for level, (f_a, gt_f_wrap_b) in enumerate(gt_f_pairs):
        N, C, H, W = gt_f_wrap_b.shape
        sel_a_idx = sel_a_indices[:, 2 - level, :].view(N, M).detach()
        gt_f_wrap_b_weighted = F.conv2d(gt_f_wrap_b.view(N * C, 1, H, W), kernel, padding=3).view(N, C, H, W)
        f_a_select = batched_index_select(f_a.view(N, C, H * W), 2, sel_a_idx)
        gt_f_wrap_b_weighted_select = batched_index_select(gt_f_wrap_b_weighted.view(N, C, H * W), 2, sel_a_idx)
        e = f_a_select - gt_f_wrap_b_weighted_select
        feature_map_loss += torch.mean(e * e)
    return feature_map_loss


""" Do Training --------------------------------------------------------------------------------------------------------
"""
time_start = datetime.datetime.now()
itr = 0
for epoch in range(0, train_params.MAX_EPOCHS):
    try:

        # Iterate the
        for train_batch_idx, train_dict in tqdm(enumerate(train_loader),
                                                total=len(train_loader),
                                                desc='Train epoch = %d, lr=%f' % (epoch, dl_util.get_learning_rate(optimizer)),
                                                ncols=100, leave=False):
            itr += 1

            # Switch to Train model
            net.train()

            # zero the parameter gradients
            optimizer.zero_grad()

            # Pre-process the variables
            # Generate 3 pyramid level with each random select 2000 samples
            I_a, d_a, sel_a_indices, K, I_b, q_gt, t_gt, se3_gt, T_gt = ba_tracknet_preprocess(train_dict, 3, 2000)
            I_a = I_a.cuda()
            d_a = d_a.cuda()
            sel_a_indices = sel_a_indices.cuda()
            K = K.cuda()
            I_b = I_b.cuda()
            q_gt = q_gt.cuda()
            t_gt = t_gt.cuda()
            se3_gt = se3_gt.cuda()
            T_gt = T_gt.cuda()

            # Forward
            pred_pose, gt_f_pairs, init_T, lambda_weight, flow_list, gt_f_patch_pairs = net.train_feed(I_a, d_a, sel_a_indices, K, I_b, se3_gt, epoch)

            # Compute Losses
            R_gt, t_gt = se3_exp(se3_gt)
            rot_error, trans_error = pose_loss(pred_pose, q_gt, t_gt, np.linspace(0.5, 1.3, num=9))
            reg_loss = torch.tensor(0.0)#feature_reg_loss(gt_f_pairs)

            flow, flow_gt = flow_list[-1][-1]
            # mask out faraway wrapping points
            flow_norm = torch.norm(flow - flow_gt, p=2, dim=3)
            flow_mask = torch.lt(flow_norm, 2.828)
            flow_accu = torch.mean(torch.masked_select(flow_norm, flow_mask))

            total_loss = 0.5 * rot_error + trans_error #+ 0.1 * flow_accu

            # Compute Accuracy
            T = pred_pose[-1][-1]
            R, t = T[:, :3, :3], T[:, :3, 3:]
            q = batched_rot2quaternion(R)
            final_q_accu = torch.mean(relative_angle(q, q_gt))
            final_t_accu = F.mse_loss(t, t_gt)

            #flow_offset = torch.masked_select(flow_norm, flow_mask)

            # Compute noise level, same measurement as Accuracy
            init_R = init_T[:, :3, :3]
            init_t = init_T[:, :3, 3:]
            init_q = batched_rot2quaternion(init_R)
            init_q_accu = torch.mean(relative_angle(init_q, q_gt))
            init_t_accu = F.mse_loss(init_t, t_gt)

            # Do backwards
            total_loss.backward()

            # Backward and optimize
            optimizer.step()

            # Log the loss and other information every LOG_STEPS mini-batches
            if train_batch_idx % train_params.LOG_STEPS == 0:
                log_dict = {
                    'Iteration': itr + 1,
                    'Epoch': epoch,
                    'Event': 'Training',
                    'Loss(Train)/rot_loss': rot_error.item(),
                    'Loss(Train)/trans_loss': trans_error.item(),
                    'Loss(Train)/reg_loss': reg_loss.item(),
                    'Loss(Train)/total_loss': total_loss.item(),
                    'Accuracy(Train)/final_rot_error': final_q_accu.item(),
                    'Accuracy(Train)/final_trans_error': final_t_accu.item(),
                    'Accuracy(Train)/init_rot_error': init_q_accu.item(),
                    'Accuracy(Train)/init_trans_error': init_t_accu.item(),
                    'Accuracy(Train)/rel_rot_error': (final_q_accu - init_q_accu).item(),
                    'Accuracy(Train)/rel_trans_error': (final_t_accu - init_t_accu).item(),
                    'Accuracy(Train)/final_flow_error': flow_accu.item(),
                    #'Histogram(Train)/flow': flow_offset

                # 'Scalar/lm_lambda_mean': torch.mean(lambda_weight[-1][-1]).item()
                }
                # if train_batch_idx % (5 * train_params.LOG_STEPS) == 0:
                #     log_dict['Net'] = net
                logger.log(log_dict)
                # print('Epoch', epoch, 'Itr', itr, ' Loss=', total_loss.item())

            # Save checkpoints on the end of epoch or fixed iteration steps
            if (itr + 1) % train_params.CHECKPOINT_STEPS == 0:
                checkpoint_file_name = os.path.join(model_checkpoint_dir, 'iter_%06d.pth.tar' % (itr + 1))
                logger.log({
                    'Iteration': itr + 1,
                    'Epoch': epoch,
                    'Event': "Check point saved to %s" % checkpoint_file_name
                })
                dl_util.save_checkpoint({
                    'log_dir': logger.log_base_dir,
                    'iteration': itr + 1,
                    'epoch': epoch + 1,
                    'net_instance': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename=checkpoint_file_name, is_best=False)

            """ Do Validating-------------------------------------------------------------------------------------------
            """
            # Validating every 2000 iteration
            if (itr + 1) % 1000 == 0:
                total_valid_batch = 0

                # Collections
                valid_q_accu = []
                valid_t_accu = []
                valid_q_init_accu = []
                valid_t_init_accu = []
                valid_q_rel_accu = []
                valid_t_rel_accu = []
                valid_flow_accu = []

                for valid_batch_idx, valid_dict in enumerate(valid_loader):
                    net.eval()

                    I_a, d_a, sel_a_indices, K, I_b, q_gt, t_gt, se3_gt, T_gt = ba_tracknet_preprocess(valid_dict, 3,
                                                                                                       1500)
                    I_a = I_a.cuda()
                    d_a = d_a.cuda()
                    sel_a_indices = sel_a_indices.cuda()
                    K = K.cuda()
                    I_b = I_b.cuda()
                    q_gt = q_gt.cuda()
                    t_gt = t_gt.cuda()
                    se3_gt = se3_gt.cuda()
                    T_gt = T_gt.cuda()

                    pred_pose, gt_f_pairs, init_T, flow_list = net.valid(I_a, d_a, sel_a_indices, K, I_b, se3_gt, epoch)

                    # Compute Accuracy
                    T = pred_pose[-1][-1]
                    R, t = T[:, :3, :3], T[:, :3, 3]
                    q = batched_rot2quaternion(R)
                    final_q_accu = torch.mean(relative_angle(q, q_gt))
                    final_t_accu = F.mse_loss(t, t_gt)
                    flow, flow_gt = flow_list[-1][-1]
                    # mask out faraway wrapping points
                    flow_norm = torch.norm(flow - flow_gt, p=2, dim=3)
                    flow_mask = torch.lt(flow_norm, 2.828)
                    flow_accu = torch.mean(torch.masked_select(flow_norm, flow_mask))
                    valid_q_accu.append(final_q_accu.item())
                    valid_t_accu.append(final_t_accu.item())
                    valid_flow_accu.append(flow_accu.item())

                    # Compute noise level, same measurement as Accuracy
                    init_R = init_T[:, :3, :3]
                    init_t = init_T[:, :3, 3]
                    init_q = batched_rot2quaternion(init_R)
                    init_q_accu = torch.mean(relative_angle(init_q, q_gt))
                    init_t_accu = F.mse_loss(init_t, t_gt)

                    valid_q_init_accu.append(init_q_accu.item())
                    valid_t_init_accu.append(init_t_accu.item())

                    valid_q_rel_accu.append((final_q_accu - init_q_accu).item())
                    valid_t_rel_accu.append((final_t_accu - init_t_accu).item())

                    total_valid_batch += 1
                    if valid_batch_idx > 100:
                        break

                # Show validation result
                mean_q_accu = np.mean(np.asarray(valid_q_accu))
                mean_t_accu = np.mean(np.asarray(valid_t_accu))
                mean_flow_accu = np.mean(np.asarray(valid_flow_accu))
                mean_init_q_accu = np.mean(np.asarray(valid_q_init_accu))
                mean_init_t_accu = np.mean(np.asarray(valid_t_init_accu))
                mean_rel_q_accu = np.mean(np.asarray(valid_q_rel_accu))
                mean_rel_t_accu = np.mean(np.asarray(valid_t_rel_accu))

                valid_log_dict = {
                    'Iteration': itr + 1,
                    'Epoch': epoch,
                    'Event': 'Validating',
                    'Accuracy(Valid)/final_rot_error': mean_q_accu,
                    'Accuracy(Valid)/final_trans_error': mean_t_accu,
                    'Accuracy(Valid)/init_rot_error': mean_init_q_accu,
                    'Accuracy(Valid)/init_trans_error': mean_init_t_accu,
                    'Accuracy(Valid)/rel_rot_error': mean_rel_q_accu,
                    'Accuracy(Valid)/rel_trans_error': mean_rel_t_accu,
                    'Accuracy(Valid)/final_flow_error': mean_flow_accu,
                }
                logger.log(valid_log_dict)

    finally:

        # Save the checkpoints, if any error happened
        checkpoint_file_name = os.path.join(model_checkpoint_dir, 'iter_%06d.pth.tar' % (itr + 1))

        dl_util.save_checkpoint({
            'log_dir': logger.log_base_dir,
            'iteration': itr + 1,
            'epoch': epoch + 1,
            'net_instance': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=checkpoint_file_name, is_best=False)

        warnings.warn('[Error] Save checkpoint to ' + checkpoint_file_name)