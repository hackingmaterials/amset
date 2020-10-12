import matplotlib.pyplot

from amset.plot.base import (  # noqa
    BaseMeshPlotter,
    BaseTransportPlotter,
    amset_base_style,
    revtex_style,
)


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
