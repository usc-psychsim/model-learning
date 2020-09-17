import colorsys
import numpy as np
import matplotlib

matplotlib.use('Agg')  # for linux
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from model_learning.util.io import get_file_name_without_extension, get_file_changed_extension

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

TITLE_FONT_SIZE = 12


def plot_evolution(data, labels, title, colors=None, output_img=None, x_label='', y_label='', show=False):
    """
    Plots the given data, assumed to be a collection of variables evolving over time.
    :param np.ndarray data: the data to be plotted, in the shape (num_variables, time).
    :param list[str] labels: the list of the variables' labels.
    :param str title: the title of the plot.
    :param np.ndarray or None colors: an array of shape (num_variables, 3) containing colors for each variable in the
    [R, G, B] normalized format ([0-1]). If `None`, colors will be automatically generated.
    :param str output_img: the path to the image on which to save the plot. None results in no image being saved.
    :param str x_label: the label of the X axis.
    :param str y_label: the label of the Y axis.
    :param bool show: whether to show the plot on the screen.
    :return:
    """
    assert len(labels) == data.shape[0], 'Number of given labels does not match data size!'
    assert colors is None or len(colors) == data.shape[0], 'Number of given colors does not match data size!'

    # saves to CSV data-file
    np.savetxt(get_file_changed_extension(output_img, 'csv'),
               data.transpose() if len(data.shape) > 1 else data.reshape((-1, 1)),
               '%s', ',', header=','.join(labels), comments='')

    plt.figure()

    # automatically get colors
    if colors is None:
        colors = distinct_colors(len(labels))

    # plots lines for each variable
    for i in range(len(labels)):
        plt.plot(data[i], label=labels[i], color=colors[i])

    plt.xlim([0, data.shape[1] - 1])
    format_and_save_plot(plt.gca(), title, output_img, x_label, y_label, data.shape[0] > 1, True, show)


def plot_bar(data, title, colors=None, output_img=None, plot_mean=True, x_label='', y_label='',
             show_legend=True, horiz_grid=True, show=False):
    """
    Plots the given data, assumed to be a collection of key-value pairs.
    :param dict[str, float] data: the data to be plotted.
    :param str title: the title of the plot.
    :param np.ndarray or None colors: an array of shape (num_variables, 3) containing colors for each variable in the
    [R, G, B] normalized format ([0-1]). If `None`, colors will be automatically generated.
    :param str output_img: the path to the image on which to save the plot. None results in no image being saved.
    :param bool plot_mean: whether to plot a horizontal line across the bar chart denoting the mean of the values.
    :param str x_label: the label of the X axis.
    :param str y_label: the label of the Y axis.
    :param bool show_legend: whether to show a legend. If `False`, data labels will be placed on tick marks.
    :param bool horiz_grid: whether to show an horizontal grid.
    :param bool show: whether to show the plot on the screen.
    :return:
    """
    data_size = len(data)
    labels = list(data.keys())
    values = [data[key] for key in labels]

    # save to csv
    np.savetxt(get_file_changed_extension(output_img, 'csv'), np.array([values]), '%s', ',',
               header=','.join(labels), comments='')

    # automatically get colors
    if colors is None:
        colors = distinct_colors(data_size)

    # create bar chart with mean
    plt.figure(figsize=(max(8., 0.4 * data_size), 6))
    ax = plt.gca()
    ax.bar(np.arange(data_size), values, color=colors, edgecolor='black', linewidth=0.7, zorder=100)
    if plot_mean:
        ax.axhline(y=np.mean(values), label='Mean', c='black', ls='--')

    if show_legend:
        # add custom legend on the side
        plt.xticks([])
        patches = []
        for i, color in enumerate(colors):
            patches.append(mpatches.Patch(color=color, label=labels[i]))
        leg = plt.legend(handles=patches, loc='right', fancybox=False)
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_linewidth(0.8)
    else:
        # show data labels in tick marks
        plt.xticks(np.arange(data_size), labels, rotation=45, horizontalalignment='right')

    format_and_save_plot(ax, title, output_img, x_label, y_label, False, horiz_grid, show)
    plt.close()


def format_and_save_plot(ax, title, output_img=None, x_label='', y_label='',
                         show_legend=True, horiz_grid=True, show=False):
    """
    Utility function that formats a plot and saves it to a file. Also closes the current plot.
    This gives the generated plots a uniform look-and-feel across the library.
    :param ax: the plot axes to be formatted.
    :param str title: the plot's title.
    :param str output_img: the path to the image on which to save the plot. None results in no image being saved.
    :param str x_label: the label of the X axis.
    :param str y_label: the label of the Y axis.
    :param bool show_legend: whether to show the legend.
    :param bool horiz_grid: whether to show an horizontal grid.
    :param bool show: whether to show the plot on the screen.
    :return:
    """

    plt.title(title, fontweight='bold', fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    if horiz_grid:
        ax.yaxis.grid(True, which='both', linestyle='--', color='lightgrey')
    if show_legend:
        leg = plt.legend(fancybox=False)
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_linewidth(0.8)

    if output_img is not None:
        plt.savefig(output_img, pad_inches=0, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


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


def rgb_to_hex(color):
    """
    Converts the given RGB color in hexadecimal notation
    :param list[float] color: the base color in the [R, G, B] normalized format ([0-1]).
    :return:
    """
    return '#{:02x}{:02x}{:02x}'.format(*tuple(np.array(np.asarray(color) * 255, dtype=int)))
