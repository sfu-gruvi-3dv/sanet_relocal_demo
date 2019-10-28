import os
import numpy as np
import core_3dv.camera_operator as cam_opt
import core_math.transfom as trans
import frame_seq_data
import seq_data.random_sel_frames


''' Utilities for adding noise to sequences -----------------------------------------------------------------------------
'''
def add_drift_noise(seq, rot_noise_deg=10.0, displacement_dist_std=0.1):
    """
    Add gaussian noise for the pose of keyframes, here we update the noise-track with random noise to
    simulate drift error.
    :param seq: keyframe sequences, dim: (M, 3, 4), M is the number of keyframes
    :param rot_noise_deg: noise in rotation (unit: deg)
    :param displacement_dist_std: displacement factor in translation, the unit 1 is the avg. baseline among all cameras.
    :return: noise sequences with dim (M, 3, 4), displacement std.
    """
    n_frames = seq.shape[0]

    avg_frame_dist = 0
    R, t = cam_opt.Rt(seq[0])
    pre_frame_center = cam_opt.camera_center_from_Tcw(R, t)
    for frame_idx in range(1, n_frames):
        R, t = cam_opt.Rt(seq[frame_idx])
        frame_center = cam_opt.camera_center_from_Tcw(R, t)
        dist = np.linalg.norm(frame_center - pre_frame_center)
        pre_frame_center = frame_center
        avg_frame_dist += dist
    avg_frame_dist /= n_frames

    # Set the translation noise
    loc_disp_noise_sigma = displacement_dist_std                    # std. for random displacement
    disp_noise = np.random.normal(0, loc_disp_noise_sigma, size=(n_frames, 3))

    # Set the rotation noise
    rot_noise_factor = np.deg2rad(rot_noise_deg)
    rot_noise = np.random.normal(0, rot_noise_factor, size=n_frames)

    # Add random noise, here we have two track: 'ground-truth-track' and 'noise-track'
    # the 'ground-truth-track' providing ground-truth relative pose, we then add noise
    # onto relative pose, and added to the 'noise-track' in the end.

    new_seq = seq.copy()
    pre_frame = seq[0]                                      # used for ground-truth track
    pre_noise_frame = np.eye(4)                             # noise-track, init the same with first ground-truth pose
    pre_noise_frame[:3, :] = pre_frame

    for frame_idx in range(1, n_frames):
        # Ground-truth-track
        # current frame:
        T = seq[frame_idx]
        R, t = cam_opt.Rt(T)

        # previous frame:
        pre_R, pre_t = cam_opt.Rt(pre_frame)
        pre_T = np.eye(4, dtype=np.float32)
        pre_T[:3, :3] = pre_R
        pre_T[:3, 3] = pre_t

        # inv_T = cam_opt.camera_pose_inv(R, t)
        # r_angle, r_axis, _ = trans.rotation_from_matrix(inv_T)
        # print('Old Rotation:', r_angle)

        # Compute ground-truth relative pose, and add random noise to translation
        rel_T = cam_opt.relateive_pose(pre_R, pre_t, R, t)
        rel_R, rel_t = cam_opt.Rt(rel_T)
        rel_C = cam_opt.camera_center_from_Tcw(rel_R, rel_t)
        rand_C = rel_C + disp_noise[frame_idx]

        # Add random noise to rotation
        temp_T = np.eye(4, dtype=np.float32)
        temp_T[:3, :3] = rel_R
        angle, axis, _ = trans.rotation_from_matrix(temp_T)
        new_angle = rot_noise[frame_idx]
        new_axis = np.random.normal(0, 1.0, size=3)
        noise_R = trans.rotation_matrix(new_angle, new_axis)[:3, :3]

        # print('New', np.rad2deg(new_angle))

        # Build new relative transformation matrix
        new_R = np.dot(noise_R, rel_R)
        new_t = cam_opt.translation_from_center(new_R, rand_C)
        temp_T[:3, :3] = new_R
        temp_T[:3, 3] = new_t

        # Add the noise relative transformation onto noise-track
        new_T = np.dot(temp_T, pre_noise_frame)
        new_seq[frame_idx][:3, :] = new_T[:3, :]

        # Update both ground-truth-track as well as noise-track
        pre_frame = T
        pre_noise_frame = new_T

        # inv_new_T = cam_opt.camera_pose_inv(new_T[:3, :3], new_T[:3, 3])
        # r_angle, r_axis, _ = trans.rotation_from_matrix(inv_new_T)
        # print('New Rotation:', r_angle)

    return new_seq, loc_disp_noise_sigma

def add_gaussian_noise(seq, rot_noise_deg=10.0, loc_displace_factor=0.1):
    """
    Add gaussian noise for the pose of keyframes
    :param seq: keyframe sequences, dim: (M, 3, 4), M is the number of keyframes
    :param rot_noise_deg: noise in rotation (unit: deg)
    :param loc_displace_factor: displacement factor in translation, the unit 1 is the avg. baseline among all cameras.
    :return: noise sequences with dim (M, 3, 4), displacement std.
    """
    n_frames = seq.shape[0]

    avg_frame_dist = 0
    R, t = cam_opt.Rt(seq[0])
    pre_frame_center = cam_opt.camera_center_from_Tcw(R, t)
    for frame_idx in range(1, n_frames):
        R, t = cam_opt.Rt(seq[frame_idx])
        frame_center = cam_opt.camera_center_from_Tcw(R, t)
        dist = np.linalg.norm(frame_center - pre_frame_center)
        avg_frame_dist += dist
    avg_frame_dist /= n_frames

    # Set the translation noise
    loc_disp_noise_sigma = loc_displace_factor * avg_frame_dist     # std. for random displacement
    disp_noise = np.random.normal(0, loc_disp_noise_sigma, size=(n_frames, 3))

    # Set the rotation noise
    rot_noise_factor = np.deg2rad(rot_noise_deg)
    rot_noise = np.random.normal(0, rot_noise_factor, size=n_frames)

    new_seq = seq.copy()
    for frame_idx in range(1, n_frames):
        T = seq[frame_idx]
        R, t = cam_opt.Rt(T)

        # Add random noise to translation
        C = cam_opt.camera_center_from_Tcw(R, t)
        rand_C = C + disp_noise[frame_idx]

        # Add random noise to rotation
        temp_T = np.eye(4)
        temp_T[:3, :3] = R
        angle, axis, _ = trans.rotation_from_matrix(temp_T)
        new_angle = angle + rot_noise[frame_idx]
        new_axis = axis + np.random.normal(0, 0.1, size=3)
        new_R = trans.rotation_matrix(new_angle, new_axis)[:3, :3]

        new_t = cam_opt.translation_from_center(new_R, rand_C)
        new_seq[frame_idx][:3, :3] = new_R[:3, :3]
        new_seq[frame_idx][:3, 3] = new_t

    return new_seq, loc_disp_noise_sigma

''' Validate scripts ---------------------------------------------------------------------------------------------------
'''
if __name__ == '__main__':
    # SUN3D Base dir
    base_dir = '/home/luwei/mnt/Tango/ziqianb/SUN3D'

    # Select a sequences
    ori_seq_json_path = os.path.join(base_dir, 'brown_bm_2/brown_bm_2/seq.json')

    # Toggle to show 2D seq map instead of image sequences
    show_2d_path = True

    # Load the original frame and random sample subset
    ori_seq = frame_seq_data.FrameSeqData(ori_seq_json_path)
    sub_seq_list = seq_data.random_sel_frames.rand_sel_subseq_sun3d(scene_frames=ori_seq,
                                                                    trans_thres_range=0.15,
                                                                    frames_per_subseq_num=10,
                                                                    frames_range=(0.00, 0.8),
                                                                    max_subseq_num=30,
                                                                    interval_thres=2)

    # Collect the transformation matrix
    seq_T = np.stack([frame['extrinsic_Tcw'] for frame in sub_seq_list[0].frames], axis=0)
    # noise_T, rand_std_radius = add_gaussian_noise(seq_T, rot_noise_deg=8.0, loc_displace_factor=0.1)
    noise_T, rand_std_radius = add_drift_noise(seq_T, rot_noise_deg=8.0, displacement_dist_std=0.08)

    # Show 2D Seq
    if show_2d_path:
        import matplotlib.pyplot as plt
        from seq_data.plot_seq_2d import plot_array_seq_2d
        plt.figure()
        ax = plt.gca()
        plot_array_seq_2d(noise_T, plt_axes=ax, show_view_direction=True, legend='noise', color='r')
        plot_array_seq_2d(seq_T, plt_axes=ax, show_view_direction=True, legend='ori', color='b')

        # Plot displacement radius
        for frame_idx in range(1, noise_T.shape[0]):
            T = seq_T[frame_idx]
            C = cam_opt.camera_center_from_Tcw(T[:3, :3], T[:3, 3])
            circle = plt.Circle((C[0], C[2]), 3 * rand_std_radius, color=(1.0, 0.0, 0.0, 0.1))
            ax.add_patch(circle)

        ax.set_aspect('equal', adjustable='box')
        plt.show()
    else:
        from visualizer.visualizer_3d import Visualizer

        # Show 3D case
        vis = Visualizer()

        count = 0
        def keyPressEvent(obj, event):
            global seq_T
            global noise_T
            global count
            key = obj.GetKeySym()
            if key == 'Right':
                vis.clear_frame_poses()
                if count % 2 == 0:
                    for frame_idx in range(0, noise_T.shape[0]):
                        T = seq_T[frame_idx]
                        R, t = cam_opt.Rt(T)
                        vis.add_frame_pose(R, t, color=(1.0, 1.0, 1.0), camera_obj_scale=0.05)
                else:
                    for frame_idx in range(0, noise_T.shape[0]):
                        T = noise_T[frame_idx]
                        R, t = cam_opt.Rt(T)
                        vis.add_frame_pose(R, t, color=(0.5, 0.0, 0.0), camera_obj_scale=0.05)
                count += 1

        for frame_idx in range(0, noise_T.shape[0]):
            n_T = noise_T[frame_idx]
            noise_R, noise_t = cam_opt.Rt(n_T)
            vis.add_frame_pose(noise_R, noise_t, color=(0.5, 0.0, 0.0), camera_obj_scale=0.05)

            T = seq_T[frame_idx]
            R, t = cam_opt.Rt(T)
            vis.add_frame_pose(R, t, color=(1.0, 1.0, 1.0), camera_obj_scale=0.05)

        vis.bind_keyboard_event(keyPressEvent)
        vis.show()