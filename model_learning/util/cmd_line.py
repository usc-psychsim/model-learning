import argparse

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def none_or_int(value):
    """
    Converts the given string to an int value or None.
    :param str value: the string we want to convert to an int, or `"None"` to return `None`.
    :rtype: int or None
    :return: the int value corresponding to the given string, or `None` if the string is `"None"`.
    """
    return None if value.lower() == 'none' else int(value)


def str2bool(value):
    """
    Converts the given string to a boolean value to use with the argparse library.
    See: https://stackoverflow.com/a/43357954
    :param str value: the string we want to convert to a bool.
    :rtype: bool
    :return: the boolean value corresponding to the given string.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
