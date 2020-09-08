import gzip
import logging
import os
import pickle
import shutil

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def get_file_changed_extension(file, ext):
    """
    Changes the extension of the given file.
    :param str file: the path to the file.
    :param str ext: the new file extension.
    :rtype: str
    :return: the file path with the new extension.
    """
    return os.path.join(os.path.dirname(file),
                        '{}.{}'.format(get_file_name_without_extension(file), ext.replace('.', '')))


def get_file_name_without_extension(file):
    """
    Gets the file name in the given path without extension.
    :param str file: the path to the file.
    :rtype: str
    :return: the file name in the given path without extension.
    """
    file = os.path.basename(file)
    return file.replace(os.path.splitext(file)[-1], '')


def get_files_with_extension(path, extension, sort=True):
    """
    Gets all files in the given directory with a given extension.
    :param str path: the directory from which to retrieve the files.
    :param str extension: the extension of the files to be retrieved.
    :param bool sort: whether to sort list of files based on file name.
    :rtype: list[str]
    :return: the list of files in the given directory with the required extension.
    """
    file_list = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.' + extension)]
    if sort:
        file_list.sort()
    return file_list


def get_directory_name(path):
    """
    Gets the directory name in the given path.
    :param str path: the path (can be a file).
    :rtype: str
    :return: the directory name in the given path.
    """
    return os.path.basename(os.path.dirname(path))


def create_clear_dir(path, clear=False):
    """
    Creates a directory in the given path. If it exists, optionally clears the directory.
    :param str path: the path to the directory to create/clear.
    :param bool clear: whether to clear the directory if it exists.
    :return:
    """
    if clear and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)


def save_object(obj, file_path, compress_gzip=True):
    """
    Saves a binary file containing the given data.
    :param obj: the object to be saved.
    :param str file_path: the path of the file in which to save the data.
    :param bool compress_gzip: whether to gzip the output file.
    :return:
    """
    with gzip.open(file_path, 'wb') if compress_gzip else open(file_path, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_object(file_path):
    """
    Loads an object from the given file, possibly gzip compressed.
    :param str file_path: the path to the file containing the data to be loaded.
    :return: the data loaded from the file.
    """
    try:
        with gzip.open(file_path, 'rb') as file:
            return pickle.load(file)
    except OSError:
        with open(file_path, 'rb') as file:
            return pickle.load(file)


def change_log_handler(log_file, verbosity=1, fmt='[%(asctime)s %(levelname)s] %(message)s'):
    """
    Changes logger to use the given file.
    :param str log_file: the path to the intended log file.
    :param int verbosity: the level of verbosity, the higher the more severe messages will be logged.
    :param str fmt: the formatting string for the messages.
    :return:
    """
    log = logging.getLogger()
    for handler in log.handlers[:]:
        log.removeHandler(handler)
    file_handler = logging.FileHandler(log_file, 'w')
    formatter = logging.Formatter(fmt)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    log.level = logging.WARN if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
