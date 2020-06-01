import os
import shutil

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


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
