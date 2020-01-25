from pathlib import Path

import click
from click import argument, option

image_type = click.Choice(["pdf", "png", "svg"], case_sensitive=False)
path_type = click.Path(exists=True)


@click.group()
def plot():
    """
    Plot AMSET results, including scattering rates and band structures
    """
    pass


@plot.command()
@argument("filename", type=path_type)
@option("-t", "--temperature", default=0, help="temperature index [default: 0]")
@option("-d", "--doping", default=0, help="doping index [default: 0]")
@option("-l", "--line-density", default=100, help="band structure line density")
@option("-p", "--prefix", help="output filename prefix")
@option("--ymin", help="minimum yaxis limit")
@option("--ymax", help="maximum yaxis limit")
@option("--interpolation-factor", default=1, help="BoltzTraP interpolation factor")
@option("--distance-factor", default=10, help="additional interpolation of lineshape")
@option("--directory", type=path_type, help="file output directory")
@option("--image-format", default="pdf", type=image_type, help="image format")
@option("--style", help="matplotlib style specification")
def lineshape(filename, **kwargs):
    """
    Plot band structures with electron lineshape
    """
    from amset.plot.band_structure import AmsetBandStructurePlotter

    plotter = AmsetBandStructurePlotter(
        filename, print_log=True, interpolation_factor=kwargs["interpolation_factor"]
    )
    plt = plotter.get_plot(
        kwargs["doping"],
        kwargs["temperature"],
        line_density=kwargs["line_density"],
        ymin=kwargs["ymin"],
        ymax=kwargs["ymax"],
        distance_factor=kwargs["distance_factor"],
    )

    save_plot(
        plt, "band", kwargs["directory"], kwargs["prefix"], kwargs["image_format"]
    )
    return plt


@plot.command()
@argument("filename", type=path_type)
@option("-p", "--prefix", help="output filename prefix")
@option("--directory", type=path_type, help="file output directory")
@option("--image-format", default="pdf", type=image_type, help="image format")
@option(
    "--separate-rates/--no-separate-rates",
    default=False,
    help="whether to separate scattering mechanisms",
)
def rates(filename, **kwargs):
    """
    Plot scattering rates
    """
    from amset.plot.rates import AmsetRatesPlotter

    plotter = AmsetRatesPlotter(filename)
    plt = plotter.get_plot(separate_rates=kwargs["separate_rates"])

    save_plot(
        plt, "rates", kwargs["directory"], kwargs["prefix"], kwargs["image_format"]
    )
    return plt


def save_plot(plt, name, directory, prefix, image_format):

    if prefix:
        prefix = prefix + "_"
    else:
        prefix = ""
    filename = Path("{}{}.{}".format(prefix, name, image_format))

    if directory:
        filename = directory / filename

    plt.savefig(filename, format=image_format, bbox_inches="tight", dpi=400)
