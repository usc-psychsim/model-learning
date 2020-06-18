import numpy as np

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

EPS = 1e-6


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


def get_jensen_shannon_divergence(dist1, dist2):
    """
    Computes the Jensen-Shannon divergence between two probability distributions. Higher values (close to 1) mean that
    the distributions are very dissimilar while low values (close to 0) denote a low divergence, similar distributions.
    See: https://stackoverflow.com/a/40545237
    See: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    Ref: Lin J. 1991. "Divergence Measures Based on the Shannon Entropy".
        IEEE Transactions on Information Theory. (33) 1: 145-151.
    Input must be two probability distributions of equal length that sum to 1.
    :param np.ndarray dist1: the first probability distribution.
    :param np.ndarray dist2: the second probability distribution.
    :rtype: float
    :return: the divergence between the two distributions in [0,1].
    """
    assert dist1.shape == dist2.shape, 'Distribution shapes do not match'
    dist1 = np.clip(dist1, EPS, None)
    dist2 = np.clip(dist2, EPS, None)

    def _kl_div(a, b):
        return np.sum(a * np.log2(a / b))

    m = 0.5 * (dist1 + dist2)
    return 0.5 * (_kl_div(dist1, m) + _kl_div(dist2, m))


def get_pairwise_jensen_shannon_divergence(dist1, dist2):
    """
    Computes the pairwise Jensen-Shannon divergence between two discrete distributions. This corresponds to the
    un-summed JSD, i.e., the divergence according to each component of the given distributions. Summing up the returned
    array yields the true JSD between the two distributions. Getting the square root of the sum results in a true
    distance metric. Higher values (close to 1) mean that the distributions are very dissimilar while low values
    (close to 0) denote a low divergence, i.e., similar distributions.
    See: https://stackoverflow.com/a/40545237
    See: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    Ref: Endres, D. M.; J. E. Schindelin (2003). "A new metric for probability distributions".
        IEEE Transactions on Information Theory. 49 (7): 1858–1860.
    :param np.ndarray dist1: the first discrete distribution.
    :param np.ndarray dist2: the second discrete distribution.
    :return np.ndarray: the divergence between each component of the two distributions in [0,1].
    """

    def _rel_entr(a, b):
        return np.array([a[i] * np.log2(a[i] / b[i]) if a[i] > 0 and b[i] > 0 else 0 for i in range(len(a))])

    def _kl_div(a, b):
        return _rel_entr(1.0 * a / np.sum(a, axis=0), 1.0 * b / np.sum(b, axis=0))

    m = 0.5 * (dist1 + dist2)
    return 0.5 * (_kl_div(dist1, m) + _kl_div(dist2, m))
