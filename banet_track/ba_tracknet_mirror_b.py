import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from banet_track.ba_backbone_drn import drn_c_26, drn_d_54
from banet_track.ba_module import se3_exp, batched_index_select, batched_interp2d, batched_gradient, batched_pi, batched_pi_inv, batched_transpose, x_2d_coords_torch
import banet_track.ba_module as module
from core_dl.base_net import BaseNet
from core_math.transfom import quaternion_matrix
from visualizer.visualizer_2d import show_multiple_img

""" Configuration ------------------------------------------------------------------------------------------------------
"""
drn_c_26_pretrained_path = 'data/drn_c_26-ddedf421.pth'

drn_d_54_pretrained_path = 'data/drn_d_54-0e0534ff.pth'

"""  Network Definition ------------------------------------------------------------------------------------------------
"""
class AggregateUnit(nn.Module):

    def __init__(self, in_1_channels, in_2_channels, out_channels):
        super(AggregateUnit, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=(in_1_channels + in_2_channels), out_channels=128,
                      kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.SELU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.SELU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.SELU(inplace=True)
        )

    def forward(self, layer_1, layer_2):
        """
        :param layer_1:
        :param layer_2:
        :return:
        """
        layer_1_up = F.interpolate(layer_1, scale_factor=2)
        out = self.conv1(torch.cat((layer_1_up, layer_2), dim=1))
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class TrackNet(BaseNet):
    """
    Track Net based on BA-Net

    >>> track_net = TrackNet()
    >>>

    """

    def train(self, mode=True):
        super(TrackNet, self).train(mode)
        # self.backbone_net.eval()

    def set_train_batch_size(self, N):
        """
        Set the training batch size and re-cache the
        :param N: number of mini-batches in one batch
        """
        C, H, W = self.input_shape_chw
        self.n_train_batch = N
        self.x_train_2d = []
        self.x_train_2d.append(x_2d_coords_torch(N, H, W).view(N, H*W, 2))                          # Level 0: 640 x 480
        self.x_train_2d.append(x_2d_coords_torch(N, int(H/2), int(W/2)).view(N, int(H*W/4), 2))     # Level 1: 320 x 240
        self.x_train_2d.append(x_2d_coords_torch(N, int(H/4), int(W/4)).view(N, int(H*W/16), 2))    # Level 2: 160 x 120

    def set_eval_batch_size(self, N):
        """
        Set the evaluate batch size and re-cache the
        :param N: number of mini-batches in one batch
        """
        C, H, W = self.input_shape_chw
        self.n_eval_size = N
        self.x_valid_2d = []
        self.x_valid_2d.append(x_2d_coords_torch(N, H, W).view(N, H*W, 2))                          # Level 0: 640 x 480
        self.x_valid_2d.append(x_2d_coords_torch(N, int(H/2), int(W/2)).view(N, int(H*W/4), 2))     # Level 1: 320 x 240
        self.x_valid_2d.append(x_2d_coords_torch(N, int(H/4), int(W/4)).view(N, int(H*W/16), 2))    # Level 2: 160 x 120

    def __init__(self, backbone_net='drn_d_54', input_dim=(3, 240, 320), num_pyramid_features=32):
        super(TrackNet, self).__init__()
        self.input_shape_chw = input_dim

        self.level_dim_hw = [np.asarray(input_dim[1:], dtype=np.int32),
                             np.asarray(input_dim[1:], dtype=np.int32) // 2,
                             np.asarray(input_dim[1:], dtype=np.int32) // 4]

        # Configure the backbone net
        if backbone_net == 'drn_d_54':
            # Output shape (C,H,W) of each conv layer:
            # level_1: [16, 640, 480]
            # level_2: [32, 320, 240]
            # level_3: [256, 160, 120]
            # level_4: [512, 80, 60]

            # Pyramid extraction
            self.aggr_l3_l4 = AggregateUnit(in_1_channels=512, in_2_channels=256, out_channels=num_pyramid_features)
            self.aggr_l2_l3 = AggregateUnit(in_1_channels=num_pyramid_features, in_2_channels=32, out_channels=num_pyramid_features)
            self.aggr_l1_l2 = AggregateUnit(in_1_channels=num_pyramid_features, in_2_channels=16, out_channels=num_pyramid_features)

        elif backbone_net == 'drn_c_26':
            # Output shape (C,H,W) of each conv layer:
            # level_1: [16, 640, 480]
            # level_2: [32, 320, 240]
            # level_3: [64, 160, 120]
            # level_4: [128, 80, 60]

            # Pyramid extraction
            self.aggr_l3_l4 = AggregateUnit(in_1_channels=128, in_2_channels=64, out_channels=num_pyramid_features)
            self.aggr_l2_l3 = AggregateUnit(in_1_channels=num_pyramid_features, in_2_channels=32, out_channels=num_pyramid_features)
            self.aggr_l1_l2 = AggregateUnit(in_1_channels=num_pyramid_features, in_2_channels=16, out_channels=num_pyramid_features)

        # Lambda Prediction Branch, 3 pyramids use different nets, 0: highest resolution
        self.lambda_fc_0 = nn.Sequential(
            nn.Linear(in_features=num_pyramid_features, out_features=128),              # FC1
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=128, out_features=1),
        )
        self.lambda_fc_1 = nn.Sequential(
            nn.Linear(in_features=num_pyramid_features, out_features=128),  # FC1
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=128, out_features=1),
        )
        self.lambda_fc_2 = nn.Sequential(
            nn.Linear(in_features=num_pyramid_features, out_features=128),  # FC1
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=128, out_features=1),
        )

        # Init the weight with xavier
        def fill_with_xavier(m):
            if type(m) == nn.Conv2d:
                torch.nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    pass
                    # m.bias.data.fill_(0.001)
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    pass
                    # m.bias.data.fill_(0.005)
        self.apply(fill_with_xavier)

        # Load the backbone net
        if backbone_net == 'drn_d_54':
            self.backbone_net = drn_d_54(drn_d_54_pretrained_path)
        elif backbone_net == 'drn_c_26':
            self.backbone_net = drn_c_26(drn_c_26_pretrained_path)

        if torch.cuda.is_available():
            self.gpu_enabled = True
            self.cuda()

    def aggregate_pyramid_features(self, raw_feature_pyramid):
        pyramid_level_3 = self.aggr_l3_l4.forward(raw_feature_pyramid[3], raw_feature_pyramid[2])
        pyramid_level_2 = self.aggr_l2_l3.forward(pyramid_level_3, raw_feature_pyramid[1])
        pyramid_level_1 = self.aggr_l1_l2.forward(pyramid_level_2, raw_feature_pyramid[0])
        return (pyramid_level_1, pyramid_level_2, pyramid_level_3)

    def forward_features(self, input):
        N = input.shape[0]

        assert input.dtype == torch.float32
        assert input.shape[-3:] == self.input_shape_chw
        assert N == self.aggr_pyramid_f_a[0].shape[0]

        # Aggregate pyramid feature on frame B
        raw_pyramid_f_b = self.backbone_net.forward(input)
        aggr_pyramid_f_b = self.aggregate_pyramid_features(raw_pyramid_f_b)
        return self.aggr_pyramid_f_a, aggr_pyramid_f_b, self.raw_pyramid_f_a, raw_pyramid_f_b

    def forward(self, input):
        N = input.shape[0]

        assert input.dtype == torch.float32
        assert input.shape[-3:] == self.input_shape_chw
        assert N == self.aggr_pyramid_f_a[0].shape[0]

        # Aggregate pyramid feature on frame B
        aggr_pyramid_f_b = self.aggregate_pyramid_features(self.backbone_net.forward(input))
        alpha = torch.tensor([1e-4, 1e-4, 1e-4, 0.0, 0.0, 0.0]).repeat(N).view((N, 6)).detach() # Init a se(3) vector

        for level in [2, 1, 0]:

                (level_H, level_W) = self.level_dim_hw[level]

                M = self.sel_a_indices.shape[2]                                                 # number of selected pts

                # Features on current level
                f_a = self.aggr_pyramid_f_a[level]
                f_b = aggr_pyramid_f_b[level]
                f_b_grad = batched_gradient(f_b)                                                # dim: (N, 2*C, H, W)

                # Resize and Rescale the depth and the intrinsic matrix
                rescale_ratio = 1.0 / math.pow(2, level)
                level_K = rescale_ratio * self.K.detach()                                       # dim: (N, 3, 3)
                level_d_a = F.interpolate(self.d_a, scale_factor=rescale_ratio).detach()        # dim: (N, 1, H, W)
                sel_a_idx = self.sel_a_indices[:, level, :].view(N, M).detach()                 # dim: (N, M)

                # Cache several variables:
                x_a_2d = self.x_valid_2d[level]                                                 # dim: (N, H*W, 2)
                X_a_3d = batched_pi_inv(level_K, x_a_2d,
                                        level_d_a.view((N, level_H * level_W, 1)))
                X_a_3d_sel = batched_index_select(X_a_3d, 1, sel_a_idx)                         # dim: (N, M, 3)

                # Run iteration 3 times
                for itr in range(0, 6):
                    alpha, r, delta_norm = module.dm_levenberg_marquardt_itr(alpha, X_a_3d, X_a_3d_sel, f_a, sel_a_idx,
                                                                             level_K, f_b, f_b_grad, self.lambda_prediction)
                    print("Level:", level, " Itr", itr, " delta norm:", delta_norm)
        return alpha

    def pre_cache(self, I_a, d_a, sel_a_indices, K):
        """
        Pre cache the variable for prediction
        :param I_a: Image of frame A, dim: (N, C, H, W)
        :param d_a: Depth of frame A, dim: (N, 1, H, W)
        :param sel_a_indices: (N, 3, M)
        :param K: intrinsic matrix at level 0: dim: (N, 3, 3)
        :param I_b: Image of frame B, dim: (N, C, H, W)
        """
        if self.training:
            raise Exception("train_feed() only used in evaluate mode")

        (N, C, H, W) = I_a.shape

        # Asserts
        assert d_a.shape == (N, 1, H, W)
        assert K.shape == (N, 3, 3)
        assert sel_a_indices.dtype == torch.int64
        assert I_a.dtype == torch.float32
        assert d_a.dtype == torch.float32
        assert hasattr(self, 'x_train_2d' if self.training else 'x_valid_2d')

        self.d_a = d_a
        self.sel_a_indices = sel_a_indices
        self.K = K

        # Aggregate pyramid features
        self.raw_pyramid_f_a = self.backbone_net.forward(I_a)
        self.aggr_pyramid_f_a = self.aggregate_pyramid_features(self.raw_pyramid_f_a)

    def random_alpha(self, se3_gt, factor):
        """
        :param se3_gt: (N, 6)
        :return:
        """
        N = se3_gt.shape[0]
        se3_array = se3_gt.detach().cpu().numpy()        # dim: (N, 6)
        r_arr_var = factor*np.var(se3_array[:, :3], axis=1).ravel()
        t_arr_var = factor*np.var(se3_array[:, 3:], axis=1).ravel()
        for batch in range(N):
            r_var = np.random.normal(0, r_arr_var[batch], size=3)
            t_var = np.random.normal(0, t_arr_var[batch], size=3)
            se3_array[batch][:3] += r_var
            se3_array[batch][3:] += t_var
        return torch.from_numpy(se3_array).cuda()

    def train_feed(self, I_a, d_a, sel_a_indices, K, I_b, se3_gt, epoch):
        """
        Pre cache the variable for prediction
        :param I_a: Image of frame A, dim: (N, C, H, W)
        :param d_a: Depth of frame A, dim: (N, 1, H, W)
        :param sel_a_indices: (N, 3, M)
        :param K: intrinsic matrix at level 0: dim: (N, 3, 3)
        :param I_b: Image of frame B, dim: (N, C, H, W)
        :param se3_gt: ground truth Pose
        """
        if not self.training:
            raise Exception("train_feed() only used in training mode")

        (N, C, H, W) = I_a.shape
        I_a.requires_grad_()
        I_b.requires_grad_()

        # Ground-truth pose
        R_gt, t_gt = se3_exp(se3_gt)

        # Concate I_a and I_b
        I = torch.cat([I_a, I_b], dim=0)

        # Aggregate pyramid features
        aggr_pyramid = self.aggregate_pyramid_features(self.backbone_net.forward(I))
        aggr_pyramid_f_a = [f[:N, :, :, :] for f in aggr_pyramid]
        aggr_pyramid_f_b = [f[N:, :, :, :] for f in aggr_pyramid]

        # Init a se(3) vector and mark requires_grad = True
        # alpha = torch.tensor([1e-4, 1e-4, 1e-4, 0.0, 0.0, 0.0]).repeat(N).view((N, 6))      # dim: (N, 6)
        factor = 0.3
        alpha = module.gen_random_alpha(se3_gt, rot_angle_rfactor=1.25, trans_vec_rfactor=0.16).view((N, 6)).cuda()
        alpha.requires_grad_()
        init_alpha = alpha.detach()

        pred_se3_list = []                                                                  # (num_level: low_res to high_res, num_iter_per_level)
        gt_f_pair_list = []
        lambda_weight = []
        for level in [2]:

            pred_se3_list.append([])
            lambda_weight.append([])
            (level_H, level_W) = self.level_dim_hw[level]

            M = sel_a_indices.shape[2]                                                      # number of selected pts

            # Features on current level
            f_a = aggr_pyramid_f_a[level]
            f_b = aggr_pyramid_f_b[level]
            f_b_grad = batched_gradient(f_b)                                                # dim: (N, 2*C, H, W)

            # Resize and Rescale the depth and the intrinsic matrix
            rescale_ratio = 1.0 / math.pow(2, level)
            level_K = rescale_ratio * K.detach()                                             # dim: (N, 3, 3)
            level_d_a = F.interpolate(d_a, scale_factor=rescale_ratio).detach()              # dim: (N, 1, H, W)
            sel_a_idx = sel_a_indices[:, level, :].view(N, M).detach()                       # dim: (N, M)

            # Cache several variables:
            x_a_2d = self.x_train_2d[level]                                                  # dim: (N, H*W, 2)
            X_a_3d = batched_pi_inv(level_K, x_a_2d,
                                    level_d_a.view((N, level_H * level_W, 1)))
            X_a_3d_sel = batched_index_select(X_a_3d, 1, sel_a_idx)                          # dim: (N, M, 3)

            """ Ground-truth correspondence for Regularizer
            """
            f_C = f_a.shape[1]
            X_b_3d_gt = batched_transpose(R_gt, t_gt, X_a_3d)
            x_b_2d_gt, _ = batched_pi(level_K, X_b_3d_gt)
            x_b_2d_gt = module.batched_x_2d_normalize(float(level_H), float(level_W), x_b_2d_gt).view(N, level_H, level_W, 2)  # (N, H, W, 2)
            gt_f_wrap_b = batched_interp2d(f_b, x_b_2d_gt)
            f_a_select = batched_index_select(f_a.view(N, f_C, level_H * level_W), 2, sel_a_idx)
            gt_f_wrap_b_select = batched_index_select(gt_f_wrap_b.view(N, f_C, level_H * level_W), 2, sel_a_idx)
            gt_f_pair_list.append((f_a_select, gt_f_wrap_b_select))

            # Run iteration 3 times
            for itr in range(0, 1):
                alpha, r, delta_norm, lamb = module.dm_levenberg_marquardt_itr(alpha, X_a_3d, X_a_3d_sel, f_a, sel_a_idx,
                                                                         level_K, f_b, f_b_grad, self.lambda_prediction, level)
                # if itr == 9:
                pred_se3_list[-1].append(alpha)
                lambda_weight[-1].append(lamb)

        return pred_se3_list, gt_f_pair_list, init_alpha, lambda_weight

    def lambda_prediction(self, r, level):
        """
        Predict lambda weight for Levenberg-Marquardt update
        :param r: residual error with dim: (N, C, M)
        :param level: pyramid level used in this iteration, int
        :return: lambda weight, dim: (N, 6)
        """
        avg_r = torch.mean(torch.abs(r), dim=2)                 # (N, C)
        lambda_fc = getattr(self, 'lambda_fc_' + str(level))
        lambda_w = F.selu(lambda_fc(avg_r)) + 2.0               # (N, 6)
        return lambda_w


    def verify_features(self, I_a, d_a, K, I_b, se3_gt, x, y, title):
        """
        Extract feature pyramids f_a, f_b of I_a and I_b
        Wrap f_b to f_a
        Compute distances of a pixel in f_a with the neighbors of its corresponding pixels in f_b
        :param I_a: Image of frame A, dim: (N, C, H, W)
        :param d_a: Depth of frame A, dim: (N, 1, H, W)
        :param K: intrinsic matrix at level 0: dim: (N, 3, 3)
        :param I_b: Image of frame B, dim: (N, C, H, W)
        :param se3_gt: Groundtruth of se3, dim: (N, 6)
        :return:
        """
        import banet_track.ba_debug as debug

        (N, C, H, W) = I_a.shape
        I_a.requires_grad_()
        I_b.requires_grad_()

        # Concate I_a and I_b
        I = torch.cat([I_a, I_b], dim=0)

        # Aggregate pyramid features
        aggr_pyramid = self.aggregate_pyramid_features(self.backbone_net.forward(I))
        aggr_pyramid_f_a = [f[:N, :, :, :] for f in aggr_pyramid]
        aggr_pyramid_f_b = [f[N:, :, :, :] for f in aggr_pyramid]

        for level in [2, 1, 0]:
            (level_H, level_W) = self.level_dim_hw[level]

            # Resize and Rescale the depth and the intrinsic matrix
            rescale_ratio = 1.0 / math.pow(2, level)
            level_K = rescale_ratio * K.detach()  # dim: (N, 3, 3)
            level_d_a = F.interpolate(d_a, scale_factor=rescale_ratio).detach()  # dim: (N, 1, H, W)

            # Cache several variables:
            R, t = se3_exp(se3_gt)
            x_a_2d = self.x_valid_2d[level]  # dim: (N, H*W, 2)
            X_a_3d = batched_pi_inv(level_K, x_a_2d,
                                    level_d_a.view((N, level_H * level_W, 1)))
            X_b_3d = batched_transpose(R, t, X_a_3d)
            x_b_2d, _ = batched_pi(level_K, X_b_3d)
            x_b_2d = module.batched_x_2d_normalize(float(level_H), float(level_W), x_b_2d).view(N, level_H, level_W, 2)  # (N, H, W, 2)

            # Wrap the feature
            level_aggr_pyramid_f_b_wrap = batched_interp2d(aggr_pyramid_f_b[level], x_b_2d)
            level_x = int(x * rescale_ratio)
            level_y = int(y * rescale_ratio)
            left = level_x - debug.similar_window_offset
            left = left if left >= 0 else 0
            right = level_x + debug.similar_window_offset
            up = level_y - debug.similar_window_offset
            up = up if up >= 0 else 0
            down = level_y + debug.similar_window_offset
            batch_distance = torch.norm(aggr_pyramid_f_a[level][:, :, up:down, left:right] -     # (N, level_H, level_W)
                                        level_aggr_pyramid_f_b_wrap[:, :, level_y:level_y+1, level_x:level_x+1], 2, 1)
            show_multiple_img([{'img': I_a[0].detach().cpu().numpy().transpose(1, 2, 0), 'title': 'I_a'},
                               {'img': I_b[0].detach().cpu().numpy().transpose(1, 2, 0), 'title': 'I_b'},
                               {'img': batch_distance[0].detach().cpu().numpy(), 'title': 'feature distance', 'cmap':'gray'}],
                              title=title, num_cols=3)



        # Test ground
# track_net = TrackNet(backbone_net='drn_c_26')
# track_net.summary()
# print(track_net)
# track_net.summary()
