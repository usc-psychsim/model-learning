import argparse

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def str2bool(v):
    """
    Converts the given string to a boolean value to use with the argparse library.
    See: https://stackoverflow.com/a/43357954
    :param str v: the string we want to convert to a bool.
    :rtype: bool
    :return: the boolean value corresponding to the given string.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
