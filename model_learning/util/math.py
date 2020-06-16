import numpy as np

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def min_max_scale(array, axis=None):
    """
    Scales the given array along the given axis such that all elements lie in [-1, 1].
    :param np.ndarray array: the array to be scaled.
    :param int axis: the axis from which to get the minimum and maximum values.
    :rtype: np.ndarray
    :return: the array scaled between -1 and 1.
    """
    a_min = np.min(array, axis)
    a_ptp = np.ptp(array, axis)
    return 2 * np.true_divide((array - a_min), a_ptp, out=np.zeros_like(array), where=a_ptp != 0) - 1.
