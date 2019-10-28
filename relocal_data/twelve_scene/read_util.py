import numpy as np
import cv2

def read_12scenes_depth(png_file_path):
    depth = cv2.imread(png_file_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    depth[depth >= 65535] = 0
    return depth / 1000.0
