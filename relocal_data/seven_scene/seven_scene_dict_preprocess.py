import torch
from banet_track.ba_module import x_2d_coords_torch, batched_pi_inv, batched_transpose, batched_inv_pose, batched_pi

def recover(X_3d, scene_center):
    NL, _, H, W = X_3d.shape
    X_3d -= scene_center.view(NL, 1, 3)
    return X_3d

def preprocess(x_2d, scene_rgb, scene_depth, scene_K, scene_Tcw, scene_ori_rgb, scene_center=None):

    N, L, C, H, W = scene_rgb.shape
    _, _, _, ori_H, ori_W = scene_ori_rgb.shape

    scene_rgb = scene_rgb.view(N * L, C, H, W)
    scene_depth = scene_depth.view(N * L, 1, H, W)
    scene_K = scene_K.view(N * L, 3, 3)
    scene_Tcw = scene_Tcw.view(N * L, 3, 4)

    # generate 3D world position of scene
    d = scene_depth.view(N * L, H * W, 1)                                                     # dim (N*L, H*W, 1)
    X_3d = batched_pi_inv(scene_K, x_2d, d)                                                   # dim (N*L, H*W, 3)
    Rwc, twc = batched_inv_pose(R=scene_Tcw[:, :3, :3],
                                t=scene_Tcw[:, :3, 3].squeeze(-1))                            # dim (N*L, 3, 3), （N, 3)
    X_world = batched_transpose(Rwc.cuda(), twc.cuda(), X_3d)                                 # dim (N*L, H*W, 3)
    X_world = X_world.view(N, L * H * W, 3)                                                   # dim (N, L*H*W, 3)
    if scene_center is None:
        scene_center = torch.mean(X_world, dim=1)                                             # dim (N, 3)
    X_world -= scene_center.view(N, 1, 3)

    X_world = X_world.view(N * L, H, W, 3).permute(0, 3, 1, 2).contiguous()                   # dim (N * L, 3, H, W)
    scene_input = torch.cat((scene_rgb, X_world), dim=1)

    return scene_input.cuda(), \
           scene_ori_rgb.view((N*L, 3, ori_H, ori_W)).cuda(), \
           X_world.cuda(), \
           torch.gt(scene_depth, 1e-5).cuda().view(N * L, H, W), \
           scene_center.cuda()
