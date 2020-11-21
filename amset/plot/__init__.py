import matplotlib.pyplot
from matplotlib import rcParams

from amset.plot.base import (  # noqa
    BaseMeshPlotter,
    BaseTransportPlotter,
    amset_base_style,
    revtex_style,
)

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


def styled_plot(*style_sheets):
    """Return a decorator that will apply matplotlib style sheets to a plot.

    ``style_sheets`` is a base set of styles, which will be ignored if
    ``no_base_style`` is set in the decorated function arguments.

    The style will further be overwritten by any styles in the ``style``
    optional argument of the decorated function.

    Args:
        style_sheets (:obj:`list`, :obj:`str`, or :obj:`dict`): Any matplotlib
            supported definition of a style sheet. Can be a list of style of
            style sheets.
    """

    def decorator(get_plot):
        def wrapper(*args, fonts=None, style=None, no_base_style=False, **kwargs):

            if no_base_style:
                list_style = []
            else:
                list_style = list(style_sheets)

            if style is not None:
                if "revtex" in style:
                    list_style = [revtex_style]
                elif isinstance(style, list):
                    list_style += style
                else:
                    list_style += [style]

            if fonts is not None:
                list_style += [{"font.family": "sans-serif", "font.sans-serif": fonts}]

            matplotlib.pyplot.style.use(list_style)
            return get_plot(*args, **kwargs)

        return wrapper

    return decorator


def pretty_subplot(
    nrows,
    ncols,
    width=None,
    height=None,
    sharex=False,
    sharey=False,
    dpi=None,
    plt=None,
    gridspec_kw=None,
):
    """Get a :obj:`matplotlib.pyplot` subplot object with pretty defaults.

    Args:
        nrows (int): The number of rows in the subplot.
        ncols (int): The number of columns in the subplot.
        width (:obj:`float`, optional): The width of the plot.
        height (:obj:`float`, optional): The height of the plot.
        sharex (:obj:`bool`, optional): All subplots share the same x-axis.
            Defaults to ``True``.
        sharey (:obj:`bool`, optional): All subplots share the same y-axis.
            Defaults to ``True``.
        dpi (:obj:`int`, optional): The dots-per-inch (pixel density) for
            the plot.
        plt (:obj:`matplotlib.pyplot`, optional): A :obj:`matplotlib.pyplot`
            object to use for plotting.
        gridspec_kw (:obj:`dict`, optional): Gridspec parameters. Please see:
            :obj:`matplotlib.pyplot.subplot` for more information. Defaults
            to ``None``.

    Returns:
        :obj:`matplotlib.pyplot`: A :obj:`matplotlib.pyplot` subplot object
        with publication ready defaults set.
    """

    if width is None:
        width = rcParams["figure.figsize"][0]
    if height is None:
        height = rcParams["figure.figsize"][1]

    # TODO: Make this work if plt is already set...
    if plt is None:
        plt = matplotlib.pyplot
        fig, axes = plt.subplots(
            nrows,
            ncols,
            sharex=sharex,
            sharey=sharey,
            dpi=dpi,
            figsize=(width, height),
            facecolor="w",
            gridspec_kw=gridspec_kw,
        )

    return fig, axes


def get_figsize(nrows, ncols, width=None, height=None, wspace=0.4, hspace=0.3):
    if width is None:
        width = matplotlib.rcParams["figure.figsize"][0]
        width = width * ncols + width * wspace * (ncols - 1)

    if height is None:
        height = matplotlib.rcParams["figure.figsize"][1]
        height = height * nrows + height * hspace * (nrows - 1)

    return width, height
