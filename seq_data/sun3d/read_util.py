# This script defines the specific data load from sun3d dataset
import cv2
import numpy as np


def read_sun3d_depth(filename, min_depth_thres=1e-5):
    """
    Read depth from a sun3d depth file
    :param filename: str
    :return depth as np.float32 array
    """
    depth_pil = cv2.imread(filename, cv2.IMREAD_ANYDEPTH).astype(np.uint16)
    depth_shifted = (depth_pil >> 3) | (depth_pil << 13)
    depth_shifted = depth_shifted.astype(np.float32)
    depth_float = (depth_shifted / 1000)
    # depth_float[depth_float < min_depth_thres] = min_depth_thres
    return depth_float