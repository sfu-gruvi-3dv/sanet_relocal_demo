import numpy as np


def gradient(gray_img):
    """
    Compute the gradient of a gray-scale image
    :param gray_img: gray-scale image
    :return: gradient with dimension (h, w, 2), the (h, w, 0) is dI/dx, while (h, w, 1) is dI/dy.
    """
    h = gray_img.shape[0]
    w = gray_img.shape[1]

    # Compute the gradient
    gradient = np.zeros((h, w, 2))
    gradient[:, 1:w-1, 0] = gray_img[:, 2:] - gray_img[:, :w-2]
    gradient[1:h-1, :, 1] = gray_img[2:, :] - gray_img[:h-2, :]
    gradient = gradient.astype(np.float32)
    return gradient