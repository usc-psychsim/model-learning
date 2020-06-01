import colorsys
import numpy as np

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def distinct_colors(n):
    """
    Generates N visually-distinct colors.
    :param int n: the number of colors to generate.
    :rtype: np.ndarray
    :return: an array of shape (n, 3) with colors in the [R, G, B] normalized format ([0-1]).
    """
    return np.array([[x for x in colorsys.hls_to_rgb(i / n, .65, .9)] for i in range(n)])


def transparent_colors(n, base_color):
    """
    Generates a range of colors with varying alpha from a given base color.
    :param int n: the number of transparent colors to generate.
    :param list[float] base_color: the base color in the [R, G, B] normalized format ([0-1]).
    :rtype: np.ndarray
    :return: an array of shape (n, 4) with colors with varying alpha in the [R, G, B, A] normalized format ([0-1]).
    """
    return np.array([base_color[:3] + [i / (n - 1)] for i in range(n)])
