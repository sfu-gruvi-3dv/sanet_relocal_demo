import numpy as np
import cv2
from img_proc.basic_proc import gradient
import matplotlib.pyplot as plt
from frame_seq_data import FrameSeqData
import core_3dv.camera_operator as cam_opt


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_32FC3).var()


def convert_rel_vo(seq:FrameSeqData, ref_T):
    for frame_idx in range(0, len(seq)):
        frame = seq.get_frame(frame_idx)
        Tcw = seq.get_Tcw(frame)
        rel_T = cam_opt.relateive_pose(ref_T[:3, :3], ref_T[:3, 3], Tcw[:3, :3], Tcw[:3, 3])
        frame['extrinsic_Tcw'] = rel_T


def select_gradient_pixels(img, depth, grad_thres=0.04, depth_thres=1e-1, max_points=None, visualize=False):
    h, w = img.shape[0], img.shape[1]
    grad = gradient(img) / 2.0
    grad_norm = np.linalg.norm(grad, axis=2)
    mask = np.logical_and(grad_norm > grad_thres, depth > depth_thres)
    sel_index = np.asarray(np.where(mask.reshape(h*w)), dtype=np.int).ravel()
    np.random.shuffle(sel_index)
    sel_index = sel_index.ravel()
    max_points = sel_index.shape[0] if max_points is not None and max_points > sel_index.shape[0] else max_points
    sel_index = sel_index if max_points is None else sel_index[:max_points]

    # Visualize
    if visualize:
        selected_mask = np.zeros((h * w), dtype=np.float32)
        selected_mask[sel_index] = 1.0
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(img, cmap='gray')
        axes[1].imshow(selected_mask.reshape(h, w), cmap='gray')
        plt.show()

    return sel_index


# # Test
# from core_io.depth_io import load_depth_from_png
# img_path = '/home/luweiy/models/rgbd_dataset/rgbd_dataset_freiburg2_large_no_loop/rgb/1311875813.769538.png'
# depth_path = '/home/luweiy/models/rgbd_dataset/rgbd_dataset_freiburg2_large_no_loop/depth/1311875813.766504.png'
#
# depth = load_depth_from_png(depth_path, div_factor=5000.0)
# img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
# sel_idx = select_gradient_pixels(img, depth, grad_thres=0.02, visualize=True, max_points=1024)
# print(sel_idx)