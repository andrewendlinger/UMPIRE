"""
Most of the utils functions are used for plotting inside the jupyter-notebooks.

Available functions:

1. plot_colorbar

2. plot_image_series

3. plot_4d
"""
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_colorbar(figure, axis, data):
    """Appends colorbar to axis and scales it according to data.

    Requires the following imports:

        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable

    'cmap' arg. deprecated, use arg. 'mappable' instead:
        img = ax.imshow(...)
        plot_colorbar(..., mappable=img)
    """
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    norm = plt.Normalize(np.min(data), np.max(data))
    figure.colorbar(
        cm.ScalarMappable(norm=norm, cmap=axis.get_children()[0].get_cmap()),
        cax=cax,
        orientation="vertical",
    )


def plot_image_series(
    arrays,
    label_list,
    nrows=1,
    cmap="plasma",
    plot_func=None,
    normalize=False,
    **subplot_kwrags
):
    """Plots series of images into subplots, optionally into multiple rows"""

    def pad_or_truncate(some_list, target_len):
        return some_list[:target_len] + [""] * (target_len - len(some_list))

    ncols = len(arrays) // nrows

    if len(label_list) != nrows * ncols:
        label_list = pad_or_truncate(label_list, nrows * ncols)

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, **subplot_kwrags)

    if normalize:
        global_min, global_max = np.min(arrays), np.max(arrays)

    for ax, arr, label in zip(axs.flat, arrays, label_list):
        if plot_func:
            plot_func(ax, arr)
            plot_colorbar(fig, ax, arr)
        else:
            if normalize:
                ax.imshow(arr, cmap=cmap, vmin=global_min, vmax=global_max)
                plot_colorbar(fig, ax, [global_min, global_max])
            else:
                ax.imshow(arr, cmap=cmap)
                plot_colorbar(fig, ax, arr)
        ax.axis("off")
        ax.set_title(label)

    fig.tight_layout()


def plot_4d(data, **scatter_kwargs):
    """4D plot = 3D colorcoded matrix plot

    https://stackoverflow.com/questions/14995610/how-to-make-a-4d-plot-with-matplotlib-using-arbitrary-data
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    mask = data < 1e5
    idx = np.arange(int(np.prod(data.shape)))
    x, y, z = np.unravel_index(idx, data.shape)
    ax.scatter(
        x,
        y,
        z,
        c=data.flatten(),
        s=10.0 * mask,
        edgecolor="face",
        alpha=0.5,
        marker="o",
        cmap="magma",
        linewidth=0,
    )
    plt.tight_layout()
    return ax
