import matplotlib.pyplot as plt
import numpy as np
import math

def show_histogram(data, bins=100, title=None, x_axis_label=None, y_axis_label=None):
    """
    Show the histogram of the data
    :param data: data with ND-array
    :param bins: num of bins in the histogram
    :param title: figure title
    :param x_axis_label: x axis label
    :param y_axis_label: y axis label
    """
    fig, axes = plt.subplots()
    n, bins, patches = axes.hist(data, bins=bins, normed=1, facecolor='green', alpha=0.75)
    if title is not None:
        axes.set_title(title)
    if x_axis_label is not None:
        axes.set_xlabel(x_axis_label)
    if y_axis_label is not None:
        axes.set_ylabel(y_axis_label)
    max_v = np.max(data)
    min_v = np.min(data)
    axes.set_xlim([min_v, max_v])
    axes.grid()
    plt.show()


def show_multiple_img(img_lists, title=None, num_cols=4, figsize=(16, 9), show=True):
    """
    :param img_lists:
                [{'img': img, 'title':'Image Title', 'cmap': None},...]
                where the cmp could be 'gray', 'jet' etc., see the imshow() in matplotlib for reference
    :param title: Super title of the figure
    :param num_cols: number of the column in this figure

    Example:
    >>>     show_multiple_img([{'img': gray_img, 'title': 'rgb'},
    >>>                        {'img': depth, 'title': 'depth', 'cmap': 'jet'},
    >>>                        {'img': normal2rgb(surface_normal), 'title': 'normal'}], title='Preview', num_cols=2)
    """

    len_figures = len(img_lists)
    rows = int(math.ceil(len_figures / num_cols))
    cols = num_cols

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    if title is not None:
        fig.suptitle(title)

    for i in range(rows):
        for j in range(cols):
            idx = i * num_cols + j
            if idx > len_figures - 1:
                break

            if rows > 1:
                ax = axs[i, j]
            else:
                ax = axs[j]

            img = img_lists[idx]['img']
            sub_title = img_lists[idx]['title'] if 'title' in img_lists[idx] else None
            cmap_option = img_lists[idx]['cmap'] if 'cmap' in img_lists[idx] else None
            if cmap_option is None:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap_option)
            if sub_title is not None:
                ax.title.set_text(sub_title)
    plt.tight_layout()
    if show:
        plt.show()


def normal2rgb(surface_normal):
    """
    Remapping the surface normal to the RGB map
    :param surface_normal: surface normal map
    :return: rgb visualization image
    """
    return (surface_normal + 1.0) / 2.0