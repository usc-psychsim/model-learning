import colorsys
import numpy as np
import matplotlib.pyplot as plt

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
    :param str output_img:
    :param str output_img: the path to the image on which to save the plot. None results in no image being saved.
    :param str x_label: the label of the X axis.
    :param str y_label: the label of the Y axis.
    :param bool show: whether to show the plot on the screen.
    :return:
    """
    plt.figure()

    assert len(labels) == data.shape[0], 'Number of given labels does not match data size!'
    assert colors is None or len(colors) == data.shape[0], 'Number of given colors does not match data size!'

    # automatically get colors
    if colors is None:
        colors = distinct_colors(len(labels))

    # plots lines for each variable
    for i in range(len(labels)):
        plt.plot(data[i], label=labels[i], color=colors[i])

    plt.xlim([0, data.shape[1] - 1])
    format_and_save_plot(plt.gca(), title, output_img, x_label, y_label, data.shape[0] > 1, True, show)


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
