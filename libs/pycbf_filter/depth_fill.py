import numpy as np
import sys
import math
import cv2
sys.path.append('./libs/pycbf_filter/build')
import pycbf_filter


def fill_depth_cross_bf(img: np.ndarray,
                        depth: np.ndarray,
                        space_sigma=[12, 5, 8],
                        range_sigma=[0.2, 0.08, 0.02],
                        max_depth_thres=None):
    """
    Fill the depth holes with cross-bilateral filter
    :param img: gray_scale image with uint8
    :param depth: depth map with meter unit
    :param space_sigma: (optional) sigmas for the spacial gaussian term.
    :param range_sigma: (optional) sigmas for the intensity gaussian term.
    :param max_depth_thres: (optional) the maximum depth threshold
    :return: refined depth
    """
    assert depth.dtype == np.float32

    H, W = depth.shape

    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    if gray_img.dtype == np.float32:
        gray_img = (gray_img * 255).astype(np.uint8)

    num_scales = len(space_sigma)
    space_sigma = np.asarray(space_sigma, dtype=np.double).reshape(num_scales, 1)
    range_sigma = np.asarray(range_sigma, dtype=np.double).reshape(num_scales, 1)

    invalid_mask = depth < 1e-4
    if max_depth_thres is not None:
        max_flag = depth > max_depth_thres
        invalid_mask = np.logical_or(invalid_mask, max_flag) 
        max_depth = min(np.max(depth), max_depth_thres)
    else:
        max_depth = np.max(depth)

    # convert the depth image to uint8.
    depth_arr = depth / max_depth
    depth_arr = np.clip(depth_arr, 0.0, 1.0)
    depth_arr = (depth_arr * 255.0).astype(np.uint8)

    # Convert to Column-Major array
    depth_arr = np.ascontiguousarray(depth_arr.T).reshape((H, W))
    invalid_mask = np.ascontiguousarray(invalid_mask.T).reshape((H, W))
    gray_img = np.ascontiguousarray(gray_img.T).reshape((H, W))

    # run cbf filter
    depth_arr = pycbf_filter.cbf_filter(depth_arr, gray_img, invalid_mask, space_sigma, range_sigma)

    # convert to meter metric
    depth_arr = depth_arr.astype(np.float32) * max_depth / 255.0
    depth_arr = depth_arr.reshape(W, H).T

    return np.ascontiguousarray(depth_arr)

if __name__ == "__main__":
    
    def read_7scenese_depth(png_file_path):
        depth = cv2.imread(png_file_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth[depth >= 65535] = 0
        return depth / 1000.0

    img_path = '/home/luwei/Desktop/nyu/frame-000002.color.png'
    depth_path = '/home/luwei/Desktop/nyu/frame-000002.depth.png'

    import cv2
    import matplotlib.pyplot as plt

    img = cv2.imread(img_path).astype(np.float32) / 255.0
    depth = read_7scenese_depth(depth_path)

    detph_refine = fill_depth_cross_bf(img, depth, max_depth_thres=5.0)
    plt.imshow(detph_refine, cmap='jet')
    plt.show()

