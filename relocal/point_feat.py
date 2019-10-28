import torch
import torch.nn as nn
import torch.nn.functional as F


def valid_avg_pool(tensor, valid_mask, kernel_size):
    valid_mask = valid_mask.float()

    N, C, H, W = tensor.shape
    out_H = H // kernel_size
    out_W = W // kernel_size

    tensor_patch = F.unfold(
        tensor,
        kernel_size=kernel_size,
        stride=kernel_size
    ).view(N, C, -1, out_H, out_W)

    valid_mask_patch = F.unfold(
        valid_mask,
        kernel_size=kernel_size,
        stride=kernel_size
    ).view(N, C, -1, out_H, out_W)

    count = torch.sum(valid_mask_patch.float(), dim=2)
    pooled_tensor = torch.sum(tensor_patch * valid_mask_patch.float(), dim=2) / \
                    torch.where(torch.le(count, 1e-5), torch.full(count.shape, 1e6).to(tensor.device), count)   # (N, 3, out_H, out_W)

    pooled_mask = torch.gt(count, 1e-3)

    return pooled_tensor, pooled_mask[:, 0, :, :]


def extract_points_validpool(scene_coords, scene_valid_mask, conv_feat, pool_kernel_size=2, point_chw_order=False):
    avg_scene_coords, pooled_mask = valid_avg_pool(scene_coords, scene_valid_mask, kernel_size=pool_kernel_size)
    if not point_chw_order:
        avg_scene_coords = avg_scene_coords.permute(0, 2, 3, 1)
        N, H, W, D = avg_scene_coords.shape
    else:
        N, D, H, W = avg_scene_coords.shape

    assert H == conv_feat.shape[2]
    assert W == conv_feat.shape[3]

    avg_scene_coords = avg_scene_coords.view((N, 3, H * W)) if point_chw_order else avg_scene_coords.view(
        (N, H * W, 3))
    C = conv_feat.shape[1]
    conv_feat = conv_feat.view((N, C, H * W))
    pooled_mask = pooled_mask.view((N, H * W))

    return avg_scene_coords, conv_feat, pooled_mask


def extract_points(scene_coords, conv_feat, pool_kernel_size=2, point_chw_order=False):
    avg_scene_coords = F.avg_pool2d(scene_coords, kernel_size=pool_kernel_size).detach()
    if not point_chw_order:
        avg_scene_coords = avg_scene_coords.permute(0, 2, 3, 1)
        N, H, W, D = avg_scene_coords.shape
    else:
        N, D, H, W = avg_scene_coords.shape

    assert H == conv_feat.shape[2]
    assert W == conv_feat.shape[3]

    avg_scene_coords = avg_scene_coords.view((N, 3, H * W)) if point_chw_order else avg_scene_coords.view(
        (N, H * W, 3))
    C = conv_feat.shape[1]
    conv_feat = conv_feat.view((N, C, H * W))

    return avg_scene_coords, conv_feat
