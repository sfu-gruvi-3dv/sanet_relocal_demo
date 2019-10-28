import datetime
import os
import sys
import warnings
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
from dataset.ours.pytorch_dataset import ImagePairDataset, ba_tracknet_preprocess
from core_dl.train_params import TrainParameters
from core_dl.logger import Logger
import core_dl.module_util as dl_util

from banet_track.ba_tracknet import TrackNet
from banet_track.ba_module import se3_exp, batched_rot2quaternion, quaternion_dist, batched_rot2angle

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
checkpoint_path = None
checkpoint = None if (checkpoint_path is None or not os.path.exists(checkpoint_path)) \
    else dl_util.load_checkpoints(checkpoint_path)

# Setup log dir
log_base_dir = os.path.join(workspace_dir, 'logs')
log_comment_msg = 'Iinit_itr1_f96_alllevel_b4_no_outside_points_quatloss'
if log_base_dir is not None:
    # Setup the logger if dir has provided
    logger = Logger(base_dir=log_base_dir, log_types='tensorboard', comment=log_comment_msg, continue_log=False)
    if train_params.VERBOSE_MODE:
        print('Log dir: %s' % logger.log_base_dir)
else:
    logger = None

dev_id = 0

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
                                           batch_size=train_params.BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=train_params.TORCH_DATALOADER_NUM)

if train_params.VERBOSE_MODE:
    print("[Dataset Overview] ----------------------------------------------------------------------------------------")
    print("Traing %d items, Valid: %d items" % (len(train_set), 0))


""" Initialize Network -------------------------------------------------------------------------------------------------
"""
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.cuda.set_device(dev_id)
net = TrackNet(backbone_net='drn_c_26', input_dim=(3, 480, 640), num_pyramid_features=96)
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
                     'Accuracy/final_rot_error',
                     'Accuracy/final_trans_error',
                     'Accuracy/init_rot_error',
                     'Accuracy/init_trans_error',
                     'Accuracy/rel_rot_error',
                     'Accuracy/rel_trans_error',
                     'Scalar/lm_lambda_mean'
                     # 'Histogram(Train)/Mu',
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

def feature_region_reg_loss(gt_f_pairs):
    inp = np.zeros((15, 15), dtype=np.float32)
    inp[7, 7] = 1
    gaussian_kernel = fi.gaussian_filter(inp, 3.5)
    target = torch.cuda.FloatTensor(gaussian_kernel).view(1, 15 * 15, 1)
    target = (1.0 - target / target.max()) * 2.0
    loss = 0.0
    for (f_a, gt_f_wrap_b) in gt_f_pairs:
        N, C, H, W = f_a.shape
        unfold_gt_f_wrap_b = F.unfold(gt_f_wrap_b, kernel_size=15, padding=7, stride=4).view(N, C, 15 * 15, H * W // 16)                # (N, C, 15*15, num_patches)
        unfold_f_a = F.unfold(f_a, kernel_size=1, padding=0, stride=4).view(N, C, 1, H * W // 16)                                       # (N, C, 1, num_patches)
        e = torch.norm(unfold_f_a - unfold_gt_f_wrap_b, p=2, dim=1)                                                                     # (N, 15*15, num_patches)
        meann = torch.mean(e, 1, keepdim=True)
        e = e / meann
        loss += torch.mean((target - e) ** 2)
    return loss

def relative_angle(q1, q2):
    return 180 * torch.acos(2*torch.sum(q1 * q2, dim=-1) ** 2 - 1) / np.pi

""" Visualize Dataset --------------------------------------------------------------------------------------------------
"""
# for frames_dict in train_loader:
#     I_a, d_a, sel_a_indices, K, I_b, pose_gt = ba_tracknet_preprocess(frames_dict, 3, 2000)
#     N = 6
#     img_list = []
#     for i in range(N):
#         img_list += [{'img': I_a[i, :, :, :].numpy().transpose(1, 2, 0), 'title': 'frame0'},
#                      {'img': I_b[i, :, :, :].numpy().transpose(1, 2, 0), 'title': 'frame1'}]
#     visualizer.visualizer_2d.show_multiple_img(img_list, num_cols=2)


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
            I_a, d_a, sel_a_indices, K, I_b, q_gt, t_gt, se3_gt, T_gt = ba_tracknet_preprocess(train_dict, 3, 1500)
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
            pred_pose, gt_f_pairs, init_T, lambda_weight = net.train_feed(I_a, d_a, sel_a_indices, K, I_b, se3_gt, epoch)

            # Compute Losses
            R_gt, t_gt = se3_exp(se3_gt)
            rot_error, trans_error = pose_loss(pred_pose, q_gt, t_gt, np.linspace(0.5, 1.0, num=3))
            reg_loss = feature_reg_loss(gt_f_pairs)
            total_loss = 0.5 * rot_error + trans_error

            # Compute Accuracy
            T = pred_pose[-1][-1]
            R, t = T[:, :3, :3], T[:, :3, 3:]
            q = batched_rot2quaternion(R)
            final_q_accu = torch.mean(relative_angle(q, q_gt))
            final_t_accu = F.mse_loss(t, t_gt)

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
                    'Accuracy/final_rot_error': final_q_accu.item(),
                    'Accuracy/final_trans_error': final_t_accu.item(),
                    'Accuracy/init_rot_error': init_q_accu.item(),
                    'Accuracy/init_trans_error': init_t_accu.item(),
                    'Accuracy/rel_rot_error': (final_q_accu - init_q_accu).item(),
                    'Accuracy/rel_trans_error': (final_t_accu - init_t_accu).item(),
                    # 'Scalar/lm_lambda_mean': torch.mean(lambda_weight[-1][-1]).item()
                }
                # if train_batch_idx % (5 * train_params.LOG_STEPS) == 0:
                #     log_dict['Net'] = net
                logger.log(log_dict)
                print('Epoch', epoch, 'Itr', itr, ' Loss=', total_loss.item())

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