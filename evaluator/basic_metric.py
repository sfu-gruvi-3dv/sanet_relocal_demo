# Note: Evaluation code from DeepTAM https://github.com/lmb-freiburg/deeptam
import numpy as np
import core_math.transfom as trans

def rel_rot_quaternion_deg(q1, q2):
    """
    Compute relative error (deg) of two quaternion
    :param q1: quaternion 1, (w, x, y, z), dim: (4)
    :param q2: quaternion 2, (w, x, y, z), dim: (4)
    :return: relative angle in deg
    """
    return 2 * 180 * np.arccos(np.clip(np.dot(q1, q2), -1.0, 1.0)) / np.pi


def rel_rot_angle(T1, T2):
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    q1 = trans.quaternion_from_matrix(R1)
    q2 = trans.quaternion_from_matrix(R2)
    return rel_rot_quaternion_deg(q1, q2)


def rel_distance(T1, T2):
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]
    d = np.dot(R1.T, t1) - np.dot(R2.T, t2)
    return np.linalg.norm(d)