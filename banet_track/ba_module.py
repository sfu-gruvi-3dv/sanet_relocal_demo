import torch
import torch.nn.functional as f
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import core_math.transfom as trans
import cv2
import skimage.measure
from skimage.transform import resize
from skimage import img_as_bool
from banet_track.ba_optimizer import gauss_newtown_update, levenberg_marquardt_update, batched_mat_inv
from visualizer.visualizer_2d import show_multiple_img
# sys.path.extend(['/opt/eigency', '/opt/PySophus'])
# from sophus import SE3


""" Utilities ----------------------------------------------------------------------------------------------------------
"""
def batched_x_2d_normalize(h, w, x_2d):
    """
    Convert the x_2d coordinates to (-1, 1)
    :param x_2d: coordinates mapping, (N, H * W, 2)
    :return: x_2d: coordinates mapping, (N, H * W, 2), with the range from (-1, 1)
    """
    x_2d[:, :, 0] = (x_2d[:, :, 0] / (float(w) - 1.0))
    x_2d[:, :, 1] = (x_2d[:, :, 1] / (float(h) - 1.0))
    x_2d = x_2d * 2.0 - 1.0
    return x_2d


def batched_interp2d(tensor, x_2d):
    """
    [TESTED, file: valid_banet_batched_interp2d.py]
    Interpolate the tensor, it will sample the pixel in input tensor by given the new coordinate (x, y) that indicates
    the position in original image. 
    :param tensor: input tensor to be interpolated to a new tensor, (N, C, H, W)
    :param x_2d: new coordinates mapping, (N, H, W, 2) in (-1, 1), if out the range, it will be fill with zero
    :return: interpolated tensor
    """
    return f.grid_sample(tensor, x_2d)


def batched_index_select(input, dim, index):
    """
    [TESTED, file: valid_bannet_batched_index_select.py]
    :param input: Tensor with shape (N, x, x, ... x)
    :param dim:   index for the dimension to be selected
    :param index: number of M indices for the selected item in different batch, (N, M)
    """
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def se3_exp(w):
    """
    [TESTED, file: valid_banet_exp_mapping.py]
    Compute the 2-order approximate exponential mapping of lie se(3) to SE(3), batched version
    Reference: http://ethaneade.com/lie_groups.pdf (Page. 12/15)
    :param  w: lie algebra se(3) tensor, dim: (N, 6), N is the batch size, for each batch, (omega, u), where u \in R^{3}
    is translation and \omega \in R^{3} is rotation component.
    :return: T[:3, :] dim: N(N, 3, 4), where the T is a SE(3) transformation matrix
    """
    N = w.shape[0]                                              # Batches

    # Cached variables
    theta_sq = torch.sum(w[:, :3] * w[:, :3], dim=1) + 1.0e-8   # Compute the theta by sqrt(\omega^T omega), dim: (N, 1)
    theta = torch.sqrt(theta_sq)                                # dim: (N, 1)
    zeros = torch.zeros(theta.shape)                            # dim: (N, 1)
    I = torch.eye(3).repeat(N, 1, 1)                            # Create batched identity matrix, dim: (N, 3, 3)

    A = torch.sin(theta) / theta                                # dim: (N,1)
    B = (1.0 - torch.cos(theta)) / theta_sq
    C = (1.0 - A) / theta_sq

    # Compute matrix with hat operators
    o_hat = torch.stack([zeros, -w[:, 2], w[:, 1],
                         w[:, 2], zeros, -w[:, 0],
                         -w[:, 1], w[:, 0], zeros], dim=1).view((-1, 3, 3))         # Skew-symmetric mat, dim: (N, 3, 3)
    o_hat2 = torch.bmm(o_hat, o_hat)                                                # dim: (N, 3, 3)

    # Rotation and translation
    # tip: .view(-1, 1, 1) used as board-casting for scalar and matrix multiply
    R = I + A.view(-1, 1, 1) * o_hat + B.view(-1, 1, 1) * o_hat2                    # dim: (N, 3, 3)
    V = I + B.view(-1, 1, 1) * o_hat + C.view(-1, 1, 1) * o_hat2                    # dim: (N, 3, 3)
    t = torch.bmm(V, w[:, 3:].view(-1, 3, 1))                                       # t = V*u, dim: (N, 3, 1)

    # return torch.cat([R, t], dim=2)
    return R, t


def transform_mat44(R, t):
    N = R.shape[0]
    bot = torch.tensor([0, 0, 0, 1], dtype=torch.float).view((1, 1, 4)).expand(N, 1, 4)
    b = torch.cat([R, t.view(N, 3, 1)], dim=2)
    return torch.cat([b, bot], dim=1)


def se3_exp_approx_order1(w):
    """
    [TESTED, file: valid_banet_exp_mapping.py]
    Compute the 1-order approximate exponential mapping of lie se(3) to SE(3), batched version
    used for small rotation and translation case, equation:
        exp(\delta \zeta ^) = (I + \delta \zeta ^)
    Reference: see SLAM14(Page. 194)
    :param  w: lie algebra se(3) tensor, dim: (N, 6), N is the batch size, for each batch, (omega, u), where u \in R^{3}
    is translation and \omega \in R^{3} is rotation component.
    :return: T[:3, :] dim: N(N, 3, 4), where the T is a SE(3) transformation matrix
    """
    N = w.shape[0]                                              # Batches
    ones = torch.ones(N)                                        # dim: (N, 1)

    R = torch.stack([ones, -w[:, 2], w[:, 1],
                     w[:, 2], ones, -w[:, 0],
                     -w[:, 1], w[:, 0], ones], dim=1).view((-1, 3, 3))             # Skew-symmetric mat, dim: (N, 3, 3)
    t = w[:, 3:].view(-1, 3, 1)
    return R, t


def x_2d_coords_torch(n, h, w):
    x_2d = np.zeros((n, h, w, 2), dtype=np.float32)
    for y in range(0, h):
        x_2d[:, y, :, 1] = y
    for x in range(0, w):
        x_2d[:, :, x, 0] = x
    return torch.Tensor(x_2d)


""" Jacobin Mat Computation --------------------------------------------------------------------------------------------
"""
def J_camera_pose(X_3d, K):
    """
    [TESTED] with numeric, when transformation is Identity Mat, other transformation has problem.
    Compute the Jacobin of Camera pose
    :param X_3d: 3D Points Position, dim: (N, M, 3), N is the batch size, M is the number sampled points
    :param fx: focal length on x dim (float32)
    :param fy: focal length on y dim (float32)
    :return: Jacobin Mat Tensor with Dim (N, M*2, 6) where the (M*2, 6) represent the Jacobin matrix and N is the batches
    """
    N = X_3d.shape[0]                       # number of batches
    M = X_3d.shape[1]                       # number of samples
    fx, fy = K[:, 0:1, 0], K[:, 1:2, 1]

    inv_z = 1 / X_3d[:, :, 2]               # 1/Z
    x_invz = X_3d[:, :, 0] * inv_z          # X/Z
    y_invz = X_3d[:, :, 1] * inv_z          # Y/Z
    xy_invz = x_invz * y_invz
    J_00 = - fx * xy_invz                   # J[0, 0] = -fx * (X * Y)/Z^2,  dim: (N, M)
    J_01 =   fx * (1.0 + x_invz ** 2)       # J[0, 1] = fx + fx * X^2 / Z^2
    J_02 = - fx * y_invz                    # J[0, 2] = - fx * Y / Z
    J_10 = - fy * (1.0 + y_invz ** 2)       # J[1, 0] = -fy - fy * Y^2/ Z^2
    J_11 =   fy * xy_invz                   # J[1, 1] = fy * (X * Y ) / Z^2
    J_12 =   fy * x_invz                    # J[1, 2] = fy * X / Z

    J_03 =   fx * inv_z                     # J[0, 3] = fx / Z
    J_04 =   torch.zeros(J_03.shape)        # J[0, 4] = 0
    J_05 = - fx * x_invz * inv_z            # J[0, 5] = - fx * X / Z^2
    J_13 =   torch.zeros(J_03.shape)        # J[1, 3] = 0
    J_14 =   fy * inv_z                     # J[1, 4] = fy / Z
    J_15 = - fy * y_invz * inv_z            # J[1, 5] = - fy * Y / Z^2

    # Stack it together
    J = torch.stack([J_00, J_01, J_02, J_03, J_04, J_05,
                     J_10, J_11, J_12, J_13, J_14, J_15], dim=2).view((N, M * 2, 6))
    return J


""" Non-linear solver --------------------------------------------------------------------------------------------------
"""
def gauss_newton(f, Jac, x0, eps=1e-4, max_itr=20, verbose=False):
    """
    Reference: https://blog.xiarui.net/2015/01/22/gauss-newton/
    :param f: residual error computation, output out dim: (N, n_f_out)
    :param Jac: jacobi matrix of input parameter, out dim: (N, n_f_out, n_f_in)
    :param x0: initial guess of parameter, dim: (N, n_f_in)
    :param eps: stop condition, when eps > norm(delta), where delta is the update vector
    :param max_itr: maximum iteration
    :param verbose: print the iteration information
    :return: x: optimized parameter
    :return: boolean: optimization converged
    """
    N = x0.shape[0]                                         # batch size
    n_f_in = x0.shape[1]                                    # input parameters

    r = f(x0)                                               # residual error r(x0), dim: (N, n_f_out)
    n_f_out = r.shape[1]

    # Iterative optimizer
    x = x0
    for itr in range(0, max_itr):

        # Compute the Jacobi with respect to the residual error
        J = Jac(x)                                          # out dim: (N, n_f_out, n_f_in)

        # Compute Update Vector: (J^tJ)^{-1} J^tR
        Jt = J.transpose(1, 2)                              # batch transpose (H,W) to (W, H), dim: (N, n_f_in, n_f_out)
        JtJ = torch.bmm(Jt, J)                              # dim: (N, n_f_in, n_f_in)
        JtR = torch.bmm(Jt, r.view(N, n_f_out, 1))          # dim: (N, n_f_in, 1)

        delta_x = torch.bmm(batched_mat_inv(JtJ), JtR).view(N, n_f_in)                              # dim: (N, n_f_in)
        delta_x_norm = torch.sqrt(torch.sum(delta_x * delta_x, dim=1)).detach().cpu().numpy()       # dim: (N, 1)

        max_delta_x_norm = np.max(delta_x_norm)
        if max_delta_x_norm < eps:
            break

        # Update parameter
        x = x - delta_x
        r = f(x)

        if verbose:
            print('[Gauss-Newton Optimizer ] itr=%d, update_norm:%f' % (itr, max_delta_x_norm))

    return x, max_delta_x_norm < eps


def batched_gradient(features):
    """
    Compute gradient of a batch of feature maps
    :param features: a 3D tensor for a batch of feature maps, dim: (N, C, H, W)
    :return: gradient maps of input features, dim: (N, ２*C, H, W), the last row and column are padded with zeros
             (N, 0:C, H, W) = dI/dx, (N, C:2C, H, W) = dI/dy
    """
    H = features.size(-2)
    W = features.size(-1)
    C = features.size(1)
    N = features.size(0)
    grad_x = (features[:, :, :, 2:] - features[:, :, :, :W - 2]) / 2.0
    grad_x = f.pad(grad_x, (1, 1, 0, 0))
    grad_y = (features[:, :, 2:, :] - features[:, :, :H - 2, :]) / 2.0
    grad_y = f.pad(grad_y, (0, 0, 1, 1))
    grad = torch.cat([grad_x.view(N, C, H, W), grad_y.view(N, C, H, W)], dim=1)
    return grad


def batched_select_gradient_pixels(imgs, depths, I_b, K, R, t, grad_thres=0.1, depth_thres=1e-4, num_pyramid=3, num_gradient_pixels=2000, visualize=False):
    """
    batch version of select gradient pixels, all operate in CPU
    :param imgs: input mini-batch gray-scale images, torch.Tensor (N, 1, H, W)
    :param depths: mini-batch depth maps, torch.Tensor (N, 1, H, W)
    :param I_b: paired images, torch.Tensor(N, C, H, W)
    :param K: camera intrinsic matrix tensor (N, 3, 3)
    :param R: rotation matrix in dimension of (N, 3, 3)
    :param t: translation vector (N, 3)
    :param grad_thres: selecting the pixel if gradient norm > gradient threshold
    :param depth_thres: selecting the pixel if depth > depth threshold
    :param num_pyramid: number of feature map pyramids used in ba_tracknet
    :param num_gradient_pixels: the number of pixels we want to select in one feature map
    :param visualize: plot selected pixels
    :return: selected indices, torch.Tensor (N, num_pyramid, num_gradient_pixels)
    """
    N, C, H, W = imgs.shape
    depths_np = depths.view(N, H, W).numpy()                                                    # (N, H, W)

    grad = batched_gradient(imgs)                                                               # (N, 2, H, W)
    grad_np = grad.numpy()
    grad_np = np.transpose(grad_np, [0, 2, 3, 1])                                               # (N, H, W, 2)
    grad_norm = np.linalg.norm(grad_np, axis=-1)                                                # (N, H, W)

    # Cache several variables:
    x_a_2d = x_2d_coords_torch(N, H, W).cpu()                                                   # (N, H*W, 2)
    X_a_3d = batched_pi_inv(K, x_a_2d.view(N, H * W, 2),
                            depths.view(N, H * W, 1))
    X_b_3d = batched_transpose(R, t, X_a_3d)
    x_b_2d, _ = batched_pi(K, X_b_3d)
    x_b_2d = batched_x_2d_normalize(float(H), float(W), x_b_2d).view(N, H, W, 2)                # (N, H, W, 2)
    I_b_wrap = batched_interp2d(I_b, x_b_2d)
    I_b_norm_wrap_np = torch.norm(I_b_wrap, p=2, dim=1).numpy()                                 # (N, H, W)

    sel_index = torch.empty((N, num_pyramid, num_gradient_pixels), device=torch.device('cpu')).long()
    for i in range(N):
        cur_H = H
        cur_W = W
        for j in range(num_pyramid):
            pixel_count = 0
            cur_grad_thres = grad_thres
            while pixel_count < num_gradient_pixels:
                cur_grad_norm = cv2.resize(grad_norm[i, :, :], dsize=(cur_W, cur_H))
                cur_depths_np = skimage.measure.block_reduce(depths_np[i, :, :], (2**j, 2**j), np.min)
                cur_I_b_norm_wrap_np = skimage.measure.block_reduce(I_b_norm_wrap_np[i, :, :], (2**j, 2**j), np.min)
                cur_mask = np.logical_and(cur_grad_norm > cur_grad_thres, cur_depths_np > depth_thres)       # (H, W)
                cur_mask = np.logical_and(cur_mask, cur_I_b_norm_wrap_np > 1e-5)
                cur_sel_index = np.asarray(np.where(cur_mask.reshape(cur_H * cur_W)), dtype=np.int)
                cur_sel_index = cur_sel_index.ravel()
                np.random.shuffle(cur_sel_index)
                num_indices = cur_sel_index.shape[0]
                start = pixel_count
                last = pixel_count + num_indices if pixel_count + num_indices < num_gradient_pixels else num_gradient_pixels
                sel_index[i, j, start:last] = torch.from_numpy(cur_sel_index[:last - start]).long()
                pixel_count += num_indices
                cur_grad_thres -= 1. / 255.
            cur_H //= 2
            cur_W //= 2

    # Visualize
    if visualize:
        img_list = [{'img': I_b[0].numpy().transpose(1, 2, 0), 'title': 'I_b'},
                    {'img': I_b_wrap[0].numpy().transpose(1, 2, 0), 'title': 'I_b_wrap_to_a'},
                    {'img': I_b_norm_wrap_np[0], 'title': 'I_b_norm_wrap_to_a', 'cmap': 'gray'},
                    {'img': imgs[0, 0].numpy(), 'title': 'I_a', 'cmap': 'gray'},
                    {'img': depths_np[0], 'title': 'd_a', 'cmap': 'gray'}]
        cur_H = H
        cur_W = W
        for i in range(num_pyramid):
            selected_mask = np.zeros((cur_H * cur_W), dtype=np.float32)
            selected_mask[sel_index[0, i, :].numpy()] = 1.0
            img_list.append({'img': selected_mask.reshape(cur_H, cur_W), 'title': 'sel_index_'+str(i), 'cmap': 'gray'})
            cur_H //= 2
            cur_W //= 2

        show_multiple_img(img_list, title='select pixels visualization', num_cols=4)

    return sel_index


""" Camera Operations --------------------------------------------------------------------------------------------------
"""
def batched_pi(K, X):
    """
    Projecting the X in camera coordinates to the image plane
    :param K: camera intrinsic matrix tensor (N, 3, 3)
    :param X: point position in 3D camera coordinates system, is a 3D array with dimension of (N, num_points, 3)
    :return: N projected 2D pixel position u (N, num_points, 2) and the depth X (N, num_points, 1)
    """
    fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
    u_x = fx * X[:, :, 0:1] / X[:, :, 2:3] + cx
    u_y = fy * X[:, :, 1:2] / X[:, :, 2:3] + cy
    u = torch.cat([u_x, u_y], dim=-1)
    return u, X[:, :, 2:3]


def batched_pi_inv(K, x, d):
    """
    Projecting the pixel in 2D image plane and the depth to the 3D point in camera coordinate.
    :param x: 2d pixel position, a 2D array with dimension of (N, num_points, 2)
    :param d: depth at that pixel, a array with dimension of (N, num_points, 1)
    :param K: camera intrinsic matrix tensor (N, 3, 3)
    :return: 3D point in camera coordinate (N, num_points, 3)
    """
    fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
    X_x = d * (x[:, :, 0:1] - cx) / fx
    X_y = d * (x[:, :, 1:2] - cy) / fy
    X_z = d
    X = torch.cat([X_x, X_y, X_z], dim=-1)
    return X


def batched_inv_pose(R, t):
    """
    Compute the inverse pose [Verified]
    :param R: rotation matrix, dim (N, 3, 3)
    :param t: translation vector, dim (N, 3)
    :return: inverse pose of [R, t]
    """
    N = R.size(0)
    Rwc = torch.transpose(R, 1, 2)
    tw = -torch.bmm(Rwc, t.view(N, 3, 1))
    return Rwc, tw


def batched_transpose(R, t, X):
    """
    Pytorch batch version of computing transform of the 3D points
    :param R: rotation matrix in dimension of (N, 3, 3)
    :param t: translation vector (N, 3)
    :param X: points with 3D position, a 2D array with dimension of (N, num_points, 3)
    :return: transformed 3D points
    """
    assert R.shape[1] == 3
    assert R.shape[2] == 3
    assert t.shape[1] == 3
    N = R.shape[0]
    M = X.shape[1]
    X_after_R = torch.bmm(R, torch.transpose(X, 1, 2))
    X_after_R = torch.transpose(X_after_R, 1, 2)
    trans_X = X_after_R + t.view(N, 1, 3).expand(N, M, 3)
    return trans_X


def batched_relative_pose(R_A, t_A, R_B, t_B):
    """
    Pytorch batch version of computing the relative pose from
    :param R_A: frame A rotation matrix
    :param t_A: frame A translation vector
    :param R_B: frame B rotation matrix
    :param t_B: frame B translation vector
    :return: Nx3x3 rotation matrix, Nx3x1 translation vector that build a Nx3x4 matrix of T = [R,t]

    Alternative way:

    R_{AB} = R_{B} * R_{A}^{T}
    t_{AB} = R_{B} * (C_{A} - C_{B}), where the C is the center of camera.

    >>> C_A = camera_center_from_Tcw(R_A, t_A)
    >>> C_B = camera_center_from_Tcw(R_B, t_B)
    >>> R_AB = np.dot(R_B, R_A.transpose())
    >>> t_AB = np.dot(R_B, C_A - C_B)
    """
    N = R_A.shape[0]

    A_Tcw = transform_mat44(R_A, t_A)
    A_Twc = batched_mat_inv(A_Tcw)

    B_Tcw = transform_mat44(R_B, t_B)

    # Transformation from A to B
    T_AB = torch.bmm(B_Tcw, A_Twc)
    return T_AB[:, :3, :]


def batched_relative_pose_mat44(R_A, t_A, R_B, t_B):
    """
    Pytorch batch version of computing the relative pose from
    :param R_A: frame A rotation matrix
    :param t_A: frame A translation vector
    :param R_B: frame B rotation matrix
    :param t_B: frame B translation vector
    :return: Nx3x3 rotation matrix, Nx3x1 translation vector that build a Nx4x4 matrix

    Alternative way:

    R_{AB} = R_{B} * R_{A}^{T}
    t_{AB} = R_{B} * (C_{A} - C_{B}), where the C is the center of camera.

    >>> C_A = camera_center_from_Tcw(R_A, t_A)
    >>> C_B = camera_center_from_Tcw(R_B, t_B)
    >>> R_AB = np.dot(R_B, R_A.transpose())
    >>> t_AB = np.dot(R_B, C_A - C_B)
    """
    N = R_A.shape[0]

    A_Tcw = transform_mat44(R_A, t_A)
    A_Twc = batched_mat_inv(A_Tcw)

    B_Tcw = transform_mat44(R_B, t_B)

    # Transformation from A to B
    T_AB = torch.bmm(B_Tcw, A_Twc)
    return T_AB

def dense_corres_a2b(d_a, K, Rab, tab, x_2d=None):
    """
    Dense correspondence from frame a to b [Verified]
    :param d_a: dim (N, H, W)
    :param K: dim (N, 3, 3)
    :param R: dim (N, 3, 3)
    :param t: dim (N, 3)
    :param x_2d: dim (N, H, W, 2)
    :return: wrapped image, dim (N, C, H, W)
    """
    N, H, W = d_a.shape

    x_a_2d = x_2d_coords_torch(N, H, W).view(N, H * W, 2) if x_2d is None else x_2d.view(N, H * W, 2)
    X_a_3d = batched_pi_inv(K, x_a_2d, d_a.view((N, H * W, 1)))
    X_b_3d = batched_transpose(Rab, tab, X_a_3d)
    x_b_2d, _ = batched_pi(K, X_b_3d)

    return x_b_2d


def wrap_b2a(I_b, d_a, K, Rab, tab, x_2d=None):
    """
    Wrap image by providing depth, rotation and translation [Verified]
    :param I_b: dim (N, C, H, W)
    :param d_a: dim (N, H, W)
    :param K: dim (N, 3, 3)
    :param Rab: dim (N, 3, 3)
    :param tab: dim (N, 3)
    :param x_2d: dim (N, H, W, 2)
    :return: wrapped image, dim (N, C, H, W)
    """
    N, C, H, W = I_b.shape

    x_a2b = dense_corres_a2b(d_a, K, Rab, tab, x_2d)
    x_a2b = batched_x_2d_normalize(H, W, x_a2b).view(N, H, W, 2)               # (N, H, W, 2)
    wrap_img_b = batched_interp2d(I_b, x_a2b)

    return wrap_img_b


""" Rotation Representation --------------------------------------------------------------------------------------------
"""
def batched_rot2quaternion(R):
    N = R.shape[0]
    diag = 1.0 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q0 = torch.sqrt(diag) / 2.0
    q1 = (R[:, 2, 1] - R[:, 1, 2]) / (4.0 * q0)
    q2 = (R[:, 0, 2] - R[:, 2, 0]) / (4.0 * q0)
    q3 = (R[:, 1, 0] - R[:, 0, 1]) / (4.0 * q0)
    q = torch.stack([q0, q1, q2, q3], dim=1)
    q_norm = torch.sqrt(torch.sum(q*q, dim=1))
    return q / q_norm.view(N, 1)


def batched_quaternion2rot(q):
    """
    [TESTED]
    :param q: normalized quaternion vector, dim: (N, 4)
    :return: rotation matrix, dim: (N, 3, 3)
    """
    N = q.shape[0]
    qw = q[:, 0]
    qx = q[:, 1]
    qy = q[:, 2]
    qz = q[:, 3]
    return torch.stack([1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw,
                        2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw,
                        2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy
                        ], dim=1).view(N, 3, 3)


def log_quaternion(q):
    u = q[:, 0:1]                                                               # (N, 1)
    v = q[:, 1:]                                                                # (N, 3)
    u = torch.clamp(u, min=-1.0, max=1.0)
    norm = torch.norm(v, 2, dim=1, keepdim=True)
    # norm = torch.clamp(norm, min=1e-8)
    return torch.acos(u) * v / norm


def exp_quaternion(log_q):
    norm = torch.norm(log_q, 2, dim=1, keepdim=True)
    # norm = torch.clamp(norm, min=1e-8)
    u = torch.cos(norm)
    v = log_q * torch.sin(norm) / norm
    return torch.cat([u, v], dim=1)


def quaternion_dist(q1, q2):
    return 1 - torch.sum(q1 * q2, dim=-1) ** 2

def batched_rot2angle(R):

    m00 = R[:, 0, 0]
    m01 = R[:, 0, 1]
    m02 = R[:, 0, 2]
    m10 = R[:, 1, 0]
    m11 = R[:, 1, 1]
    m12 = R[:, 1, 2]
    m20 = R[:, 2, 0]
    m21 = R[:, 2, 1]
    m22 = R[:, 2, 2]

    angle = torch.acos((m00 + m11 + m22 - 1)/2)
    factor = torch.sqrt((m21 - m12)**2 + (m02-m20)**2 + (m10 - m01)**2) + 1e-4
    x = (m21 - m12) / factor
    y = (m02 - m20) / factor
    z = (m10 - m01) / factor

    axis = torch.stack([x, y, z], dim=1)
    return angle, axis


""" Direct Method Core -------------------------------------------------------------------------------------------------
"""
def dm_gauss_newton_itr(alpha, X_a_3d, X_a_3d_sel, I_a, sel_a_idx, K, I_b, I_b_grad):
    """
    Special case of Direct Method at 1 iteration of gauss-newton optimization update
    :param alpha: se(3) vec: (rotation, translation), dim: (N, 6)
    :param X_a_3d: Dense 3D Point in frame A, dim: (N, H*W, 3)
    :param X_a_3d_sel: Selected semi-dense points in frame A, dim: (N, M, 3)
    :param I_a: image or feature map of frame A, dim: (N, C, H, W)
    :param sel_a_idx: selected point indices, dim: (N, M)
    :param K: Intrinsic matrix, dim: (N, 3, 3)
    :param I_b: image or feature map of frame B, dim: (N, C, H, W)
    :param I_b_grad: gradient of image or feature map of frame B, dim: (N, 2*C, H, W),
                     (N, 0:C, H, W) = dI/dx, (N, C:2C, H, W) = dI/dy
    :return: alpha: updated se(3) vector, dim: (N, 6)
    :return: e: residual error on selected point, dim: (N, M, C)
    :return: delta_norm: l2 norm of gauss-newton update vector, use for determining termination of loop
    """
    N, C, H, W = I_a.shape
    M = sel_a_idx.shape[1]

    R, t = se3_exp(alpha)

    X_b_3d = batched_transpose(R, t, X_a_3d)
    x_b_2d, _ = batched_pi(K, X_b_3d)
    x_b_2d = batched_x_2d_normalize(H, W, x_b_2d).view(N, H, W, 2)                              # (N, H, W, 2)

    # Wrap the image
    I_b_wrap = batched_interp2d(I_b, x_b_2d)

    # Residual error
    e = (I_a - I_b_wrap).view(N, C, H * W)                                                      # (N, C, H, W)
    e = batched_index_select(e, 2, sel_a_idx)                                                   # (N, C, M)

    # Compute Jacobin Mat.
    # Jacobi of Camera Pose: delta_u / delta_alpha
    du_d_alpha = J_camera_pose(X_a_3d_sel, K).view(N * M, 2, 6)                                 # (N*M, 2, 6)

    # Jacobi of Image gradient: delta_I_b / delta_u
    dI_du = batched_interp2d(I_b_grad, x_b_2d)                                                  # (N, 2*C, H, W)
    dI_du = batched_index_select(dI_du.view(N, 2 * C, H * W), 2, sel_a_idx)                     # (N, 2*C, M)
    dI_du = torch.transpose(dI_du, 1, 2).contiguous().view(N * M, 2, C)                         # (N*M, 2, C)
    dI_du = torch.transpose(dI_du, 1, 2)                                                        # (N*M, C, 2)

    # J = -dI_b/du * du/d_alpha
    J = -torch.bmm(dI_du, du_d_alpha).view(N, C * M, 6)

    # Compute the update parameters
    e = e.transpose(1, 2).contiguous().view(N, M * C)                                           # (N, M, C)
    delta, delta_norm = gauss_newtown_update(J, e)                                              # (N, 6), (N, 1)

    # Update the delta
    alpha = alpha + delta

    return alpha, e, delta_norm


def dm_levenberg_marquardt_itr(pre_T, X_a_3d, X_a_3d_sel, I_a, sel_a_idx, K, I_b, I_b_grad, lambda_func, level):
    """
    Special case of Direct Method at 1 iteration of Levenberg-Marquardt optimization update
    :param X_a_3d: Dense 3D Point in frame A, dim: (N, H*W, 3)
    :param X_a_3d_sel: Selected semi-dense points in frame A, dim: (N, M, 3)
    :param I_a: image or feature map of frame A, dim: (N, C, H, W)
    :param sel_a_idx: selected point indices, dim: (N, M)
    :param K: Intrinsic matrix, dim: (N, 3, 3)
    :param I_b: image or feature map of frame B, dim: (N, C, H, W)
    :param I_b_grad: gradient of image or feature map of frame B, dim: (N, 2*C, H, W),
                     (N, 0:C, H, W) = dI/dx, (N, C:2C, H, W) = dI/dy
    :param lambda_func: function generating \lambda vector used in Levenberg-Marquardt optimization, output dim: (N, 6)
    :param level: pyramid level used in this iteration, int
    :return: alpha: updated se(3) vector, dim: (N, 6)
    :return: e: residual error on selected point, dim: (N, M, C)
    :return: delta_norm: l2 norm of gauss-newton update vector, use for determining termination of loop
    """
    N, C, H, W = I_a.shape
    M = sel_a_idx.shape[1]

    # R, t = se3_exp(alpha)
    # print(R, t)
    pre_R = pre_T[:, :3, :3]
    pre_t = pre_T[:, :3, 3].view(N, 3, 1)
    X_b_3d = batched_transpose(pre_R, pre_t, X_a_3d)
    x_b_2d, _ = batched_pi(K, X_b_3d)
    x_b_2d = batched_x_2d_normalize(H, W, x_b_2d).view(N, H, W, 2)                              # (N, H, W, 2)

    # Wrap the image
    I_b_wrap = batched_interp2d(I_b, x_b_2d)

    # Residual error
    e = (I_a - I_b_wrap).view(N, C, H * W)                                                      # (N, C, H, W)
    e = batched_index_select(e, 2, sel_a_idx)                                                   # (N, C, M)

    # Compute Jacobin Mat.
    # Jacobi of Camera Pose: delta_u / delta_alpha
    du_d_alpha = J_camera_pose(X_a_3d_sel, K).view(N * M, 2, 6)                                 # (N*M, 2, 6)

    # Jacobi of Image gradient: delta_I_b / delta_u
    dI_du = batched_interp2d(I_b_grad, x_b_2d)                                                  # (N, 2*C, H, W)
    dI_du = batched_index_select(dI_du.view(N, 2 * C, H * W), 2, sel_a_idx)                     # (N, 2*C, M)
    dI_du = torch.transpose(dI_du, 1, 2).contiguous().view(N * M, 2, C)                         # (N*M, 2, C)
    dI_du = torch.transpose(dI_du, 1, 2)                                                        # (N*M, C, 2)

    # J = -dI_b/du * du/d_alpha
    J = -torch.bmm(dI_du, du_d_alpha).view(N, C * M, 6)

    # Compute the update parameters
    lambda_weight = lambda_func(e, level)                                                       # (N, 1)

    # Transpose the residual error to (N, M, ....)
    e = e.transpose(1, 2).contiguous().view(N, M * C)                                           # (N, M, C)
    delta, delta_norm = levenberg_marquardt_update(J, e, lambda_weight)                         # (N, 6), (N, 1)

    # Update the delta
    delta_R, delta_t = se3_exp(delta)

    # Update parameter
    new_R = torch.bmm(delta_R, pre_R)
    new_t = delta_t + torch.bmm(delta_R, pre_t.view(N, 3, 1))
    new_T = transform_mat44(new_R, new_t)

    return new_T, e, delta_norm, lambda_weight, x_b_2d


def gen_random_unit_vector():
    sum = 2.0
    while sum >= 1.0:
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        sum = x ** 2 + y ** 2
    sq = math.sqrt(1.0 - sum)
    return np.array([2.0 * x * sq, 2.0 * y * sq, 1.0 - 2.0 * sum], dtype=np.float32)

#
# def gen_random_alpha(alpha_gt, rot_angle_rfactor, trans_vec_rfactor):
#     N = alpha_gt.shape[0]
#     R, t = se3_exp(alpha_gt)
#     R_set = R.detach().cpu().numpy()
#     t_set = t.detach().cpu().numpy()
#     new_alpha = torch.zeros(N, 6)
#     for batch_idx in range(N):
#         # R = R_set[batch_idx]
#         R = np.eye(4, dtype=np.float32)
#         R[:3, :3] = R_set[batch_idx]
#         t = t_set[batch_idx]
#
#         # Add rot random noise
#         noise_axis = gen_random_unit_vector()
#         noise_angle = np.random.normal(-rot_angle_rfactor, rot_angle_rfactor)
#         # print('noise angle:', noise_angle)
#         delta_R = trans.rotation_matrix(np.deg2rad(noise_angle), noise_axis)
#         new_R = np.dot(delta_R, R)
#         old_angle, oldaxis, _ = trans.rotation_from_matrix(R)
#         new_angle, newaxis, _ = trans.rotation_from_matrix(new_R)
#         T = np.eye(4, dtype=np.float32)
#         T[:3, :3] = new_R[:3, :3]
#
#         # Add trans random noise
#         new_t = t + np.random.normal(0, trans_vec_rfactor*np.linalg.norm(t), size=(3,1))
#         T[:3, 3] = new_t.ravel()
#         T_ = SE3(T.astype(np.float64))
#         alpha_ = T_.log().ravel()
#         new_alpha[batch_idx, :3] = torch.Tensor(alpha_[3:])
#         new_alpha[batch_idx, 3:] = torch.Tensor(alpha_[:3])
#
#     return new_alpha#.cuda()
