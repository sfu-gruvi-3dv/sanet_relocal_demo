import numpy as np
import matplotlib.pyplot as plt
from frame_seq_data import FrameSeqData
import core_3dv.camera_operator as cam_opt

def plot_array_seq_2d(seq_array,
                      plt_axes=None,
                      color=None,
                      legend=None,
                      point_style='-',
                      index_range=(-1, -1),
                      show_view_direction=False,
                      arrow_color=None):
    """
    Plot frame sequences in 2D figure
    Note: use plt.show() in the end
    :param frames: frame poses with numpy array, dim: (N, 3, 4)
    :param plt_axes: matplotlib axis handle
    :param color: color tuple, e.g. (1.0, 1.0, 1.0)
    :param legend: legend name for the curve
    :param show_view_direction: show view direction in 2d
    """

    if plt_axes is not None:
        ax = plt_axes
    else:
        plt.figure()
        ax = plt.gca()

    if color is None:
        color = np.random.uniform(0, 1, size=3).tolist()

    n_frames = seq_array.shape[0]
    cam_centers = []
    view_directions = []

    avg_frame_dist = 0
    R, t = cam_opt.Rt(seq_array[0])
    pre_frame_center = cam_opt.camera_center_from_Tcw(R, t)
    for frame_idx in range(1, n_frames):
        R, t = cam_opt.Rt(seq_array[frame_idx])
        frame_center = cam_opt.camera_center_from_Tcw(R, t)
        dist = np.linalg.norm(frame_center - pre_frame_center)
        avg_frame_dist += dist
    avg_frame_dist /= n_frames

    start_idx = 0 if index_range[0] == -1 else index_range[0]
    end_idx = n_frames if index_range[1] == -1 else min(index_range[1], n_frames)

    for frame_idx in range(start_idx, end_idx):
        Tcw = seq_array[frame_idx]
        R, t = cam_opt.Rt(Tcw)
        C = cam_opt.camera_center_from_Tcw(R, t)
        view_direction = np.dot(R.T, np.asarray([0, 0, 1]))
        view_directions.append((view_direction[0], view_direction[2]))
        cam_centers.append(C)
    cam_centers = np.asarray(cam_centers)
    view_direct_len = 0.1 * avg_frame_dist
    view_directions = view_direct_len * np.asarray(view_directions)

    ax.plot(cam_centers[0, 0], cam_centers[0, 2], '*', color=color)  # First frame
    ax.plot(cam_centers[:, 0], cam_centers[:, 2], point_style, color=color, label=legend)

    if show_view_direction:
        for frame_idx in range(0, len(cam_centers)):
            ax.arrow(cam_centers[frame_idx, 0], cam_centers[frame_idx, 2],
                     view_directions[frame_idx, 0], view_directions[frame_idx, 1], width=0.01,
                     head_width=0.3 * view_direct_len, head_length=0.2 * view_direct_len,
                     fc='k', color=arrow_color[frame_idx] if arrow_color is not None else color)

    if legend is not None:
        ax.legend()
    ax.set_aspect('equal', adjustable='box')


def plot_frames_seq_2d(frames: FrameSeqData,
                       plt_axes=None,
                       color=None,
                       legend=None,
                       point_style='-',
                       index_range=(-1, -1),
                       show_view_direction=False):
    """
    Plot frame sequences in 2D figure
    Note: use plt.show() in the end
    :param frames: frame sequences
    :param plt_axes: matplotlib axis handle
    :param color: color tuple, e.g. (1.0, 1.0, 1.0)
    :param legend: legend name for the curve
    """

    if plt_axes is not None:
        ax = plt_axes
    else:
        plt.figure()
        ax = plt.gca()

    if color is None:
        color = np.random.uniform(0, 1, size=3).tolist()

    n_frames = len(frames)
    first_frame = frames.frames[0]

    avg_frame_dist = 0
    R, t = cam_opt.Rt(first_frame['extrinsic_Tcw'])
    pre_frame_center = cam_opt.camera_center_from_Tcw(R, t)

    for frame_idx in range(1, n_frames):
        R, t = cam_opt.Rt(frames.frames[frame_idx]['extrinsic_Tcw'])
        frame_center = cam_opt.camera_center_from_Tcw(R, t)
        dist = np.linalg.norm(frame_center - pre_frame_center)
        avg_frame_dist += dist
    avg_frame_dist /= n_frames

    start_idx = 0 if index_range[0] == -1 else index_range[0]
    end_idx = n_frames if index_range[1] == -1 else min(index_range[1], n_frames)

    frame_centers = []
    view_directions = []
    for frame in frames.frames[start_idx:end_idx]:
        cur_Tcw = frame['extrinsic_Tcw']
        cur_center = cam_opt.camera_center_from_Tcw(cur_Tcw[:3, :3], cur_Tcw[:3, 3])
        frame_centers.append(cur_center)

        view_direction = np.dot(R.T, np.asarray([0, 0, 1]))
        view_directions.append((view_direction[0], view_direction[2]))

    view_direct_len = 0.1 * avg_frame_dist
    view_directions = view_direct_len * np.asarray(view_directions, dtype=np.float32)
    frame_centers = np.asarray(frame_centers, dtype=np.float32)
    ax.plot(frame_centers[0, 0], frame_centers[0, 2], '*', color=color)        # First frame
    ax.plot(frame_centers[:, 0], frame_centers[:, 2], point_style, color=color, label=legend)

    if show_view_direction:
        for frame_idx in range(0, len(frame_centers)):
            ax.arrow(frame_centers[frame_idx, 0], frame_centers[frame_idx, 2],
                     view_directions[frame_idx, 0], view_directions[frame_idx, 1],
                     head_width=0.3 * view_direct_len, head_length=0.2 * view_direct_len,
                     fc='k', color=color)

    if legend is not None:
        ax.legend()
    ax.set_aspect('equal', adjustable='box')

