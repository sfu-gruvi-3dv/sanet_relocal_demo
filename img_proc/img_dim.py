import cv2
import numpy as np
from core_3dv.camera_operator import fov

def crop_by_ratio(img, crop_ratio):
    """
    Crop the image with respect to the image center by given the ratio (width to height)
    :param img: image to be cropped
    :param crop_ratio: a tuple indicates the ratio of width to the height e.g. (16, 9).
    :return: cropped image
    """
    ori_ratio_factor = float(img.shape[1]) / float(img.shape[0])                # ratio = width / height
    crop_ratio_factor = float(crop_ratio[0]) / float(crop_ratio[1])
    crop_img = None
    if ori_ratio_factor > crop_ratio_factor:
        # Crop along the x dimension
        ori_height = img.shape[0]
        new_width = int(crop_ratio_factor * float(ori_height))
        ori_center = int(img.shape[1] / 2)
        start_x = int(ori_center - 0.5 * new_width)
        end_x = int(ori_center + 0.5 * new_width)
        if len(img.shape) == 3:             # RGB
            crop_img = img[:, start_x:end_x, :]
        else:
            crop_img = img[:, start_x:end_x]
    else:
        # Crop along the y dimension
        ori_width = img.shape[1]
        new_height = int( float(ori_width) / float(crop_ratio_factor))
        ori_center = int(img.shape[0] / 2)
        start_y = int(ori_center - 0.5 * new_height)
        end_y = int(ori_center + 0.5 * new_height)
        if len(img.shape) == 3:             # RGB
            crop_img = img[start_y:end_y, :, :]
        else:
            crop_img = img[start_y:end_y, :]
    return crop_img

def crop_from_center(img, new_h, new_w):
    """
    Crop the image with respect to the center
    :param img: image to be cropped
    :param new_h: cropped dimension on height
    :param new_w: cropped dimension on width
    :return: cropped image
    """

    h = img.shape[0]
    w = img.shape[1]
    x_c = w / 2
    y_c = h / 2

    crop_img = None
    if h >= new_h and w >= new_w:
        start_x = int(x_c - new_w / 2)
        start_y = int(y_c - new_h / 2)

        if len(img.shape) > 2:
            crop_img = img[start_y: start_y + int(new_h), start_x: start_x + int(new_w), :]
        elif len(img.shape) == 2:
            crop_img = img[start_y: start_y + int(new_h), start_x: start_x + int(new_w)]

    return crop_img


def crop_by_intrinsic(img, cur_k, new_k, interp_method='bilinear'):
    """
    Crop the image with new intrinsic parameters
    :param img: image to be cropped
    :param cur_k: current intrinsic parameters, 3x3 matrix
    :param new_k: crop target intrinsic parameters, 3x3 matrix
    :return: cropped image
    """
    cur_fov_x, cur_fov_y = fov(cur_k[0, 0], cur_k[1, 1], 2 * cur_k[1, 2], 2 * cur_k[0, 2])
    new_fov_x, new_fov_y = fov(new_k[0, 0], new_k[1, 1], 2 * new_k[1, 2], 2 * new_k[0, 2])
    crop_img = None
    if cur_fov_x >= new_fov_x and cur_fov_y >= new_fov_y:
        # Only allow to crop to a smaller fov image
        # 1. Resize image
        focal_ratio = new_k[0, 0] / cur_k[0, 0]
        if interp_method == 'nearest':
            crop_img = cv2.resize(img, (int(focal_ratio * img.shape[1]), int(focal_ratio * img.shape[0])), interpolation=cv2.INTER_NEAREST)
        else:
            crop_img = cv2.resize(img, (int(focal_ratio * img.shape[1]), int(focal_ratio * img.shape[0])))
        # Crop the image with new w/h ratio with respect to the center
        crop_img = crop_from_center(crop_img, 2 * new_k[1, 2], 2 * new_k[0, 2])
    else:
        raise Exception("The new camera FOV is larger then the current.")

    return crop_img

def wrap(img, coordinate_2d, interp='bilinear'):
    """
    Wrap the image with respect to the coordinate mapping
    :param img:  the source image
    :param coordinate_2d: the mapping coordinates (refer the in source image), a 2D array with dimension of (num_points, 2)
    :param interp: interp method, 'bilinear' for bilinar interp, or 'nearest' for nearest interp.
    :return: wrapped image
    """
    h = img.shape[0]
    w = img.shape[1]
    remap_2ds = coordinate_2d.reshape((h, w, 2)).astype(np.float32)
    return cv2.remap(img, remap_2ds[:, :, 0], remap_2ds[:, :, 1], interpolation=(cv2.INTER_LINEAR if interp == 'bilinear' else cv2.INTER_NEAREST))


def undistort(img, K, k1, k2, p1=0., p2=0.):
    """
    Undistort the image according to the given intrinsics
    :param img: input image
    :param k1: distortion param
    :param k2: distortion param
    :return: undistorted image
    """
    return cv2.undistort(img, K, (k1, k2, p1, p2))