from pathlib import Path

import click
from click import argument, option

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

image_type = click.Choice(["pdf", "png", "svg"], case_sensitive=False)
kpaths = click.Choice(["pymatgen", "seekpath"], case_sensitive=False)
path_type = click.Path(exists=True)

_symprec = 0.01  # redefine symprec to avoid loading constants from file
_dos_estep = 0.01  # redefine symprec to avoid loading constants from file
_interpolation_factor = 5


def _all_or_int(value):
    if value == "all":
        return None
    elif value is None:
        return 0
    else:
        return int(value)


def _parse_kpoints(kpoints):
    return [
        [list(map(float, kpt.split())) for kpt in kpts.split(",")]
        for kpts in kpoints.split("|")
    ]


def _parse_kpoint_labels(labels):
    return [path.split(",") for path in labels.split("|")]


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
@option("-l", "--line-density", default=100.0, help="band structure line density")
@option("-p", "--prefix", help="output filename prefix")
@option("--emin", help="minimum energy limit")
@option("--emax", help="maximum energy limit")
@option("--symprec", type=float, default=_symprec, help="interpolation factor")
@option("--kpath", type=kpaths, help="k-point path type")
@option(
    "--kpoints",
    type=_parse_kpoints,
    metavar="K",
    help="manual k-points list [e.g. '0 0 0, 0.5 0 0']",
)
@option(
    "--labels",
    type=_parse_kpoint_labels,
    metavar="L",
    help=r"labels for manual kpoints [e.g. '\Gamma,X']",
)
@option("--interpolation-factor", default=1, help="BoltzTraP interpolation factor")
@option("--distance-factor", default=10.0, help="additional interpolation of lineshape")
@option("--width", default=6.0, help="figure width [default: 6]")
@option("--height", default=6.0, help="figure height [default: 6]")
@option("--directory", type=path_type, help="file output directory")
@option("--image-format", default="pdf", type=image_type, help="image format")
@option("--style", help="matplotlib style specification")
@option("--no-base-style", default=False, is_flag=True, help="don't apply base style")
def lineshape(filename, **kwargs):
    """
    Plot band structures with electron lineshape
    """
    from amset.plot.lineshape import LineshapePlotter

    plotter = LineshapePlotter(
        filename, print_log=True, interpolation_factor=kwargs["interpolation_factor"]
    )

    kpath = get_kpath(
        plotter.structure,
        mode=kwargs["kpath"],
        symprec=kwargs["symprec"],
        kpt_list=kwargs["kpoints"],
        labels=kwargs["labels"],
    )

    plt = plotter.get_plot(
        kwargs["doping"],
        kwargs["temperature"],
        line_density=kwargs["line_density"],
        emin=kwargs["emin"],
        emax=kwargs["emax"],
        distance_factor=kwargs["distance_factor"],
        width=kwargs["width"],
        height=kwargs["height"],
        style=kwargs["style"],
        no_base_style=kwargs["no_base_style"],
        kpath=kpath,
    )

    save_plot(
        plt, "band", kwargs["directory"], kwargs["prefix"], kwargs["image_format"]
    )
    return plt


@plot.command()
@argument("filename", type=path_type)
@option("-l", "--line-density", default=100.0, help="band structure line density")
@option("--emin", default=-6.0, help="minimum energy limit")
@option("--emax", default=6.0, help="maximum energy limit")
@option("--symprec", type=float, default=_symprec, help="interpolation factor")
@option("--kpath", type=kpaths, help="k-point path type")
@option(
    "--kpoints",
    type=_parse_kpoints,
    metavar="K",
    help="manual k-points list [e.g. '0 0 0, 0.5 0 0']",
)
@option(
    "--labels",
    type=_parse_kpoint_labels,
    metavar="L",
    help=r"labels for manual kpoints [e.g. '\Gamma,X']",
)
@option(
    "--interpolation-factor",
    default=_interpolation_factor,
    type=float,
    help="BoltzTraP interpolation factor",
)
@option("--energy-cutoff", type=float, help="interpolation energy cutoff in eV")
@option("--plot-dos", is_flag=True, help="whether to also plot the density of states")
@option(
    "--dos-kpoints",
    default="50",
    help="k-point length cutoff or mesh for density of states",
)
@option("--dos-estep", default=_dos_estep, help="dos energy step size")
@option("--dos-aspect", default=3.0, help="aspect ratio for the density of states")
@option(
    "--no-zero-to-efermi",
    "zero_to_efermi",
    is_flag=True,
    default=True,
    help="don't set the Fermi level to zero",
)
@option("--vbm-cbm-marker", is_flag=True, help="add a marker at the CBM and VBM")
@option("--width", default=6.0, help="figure width [default: 6]")
@option("--height", default=6.0, help="figure height [default: 6]")
@option("-p", "--prefix", help="output filename prefix")
@option("--directory", type=path_type, help="file output directory")
@option("--image-format", default="pdf", type=image_type, help="image format")
@option("--style", help="path to matplotlib style specification")
@option("--no-base-style", default=False, is_flag=True, help="don't apply base style")
def band(filename, **kwargs):
    """
    Plot interpolate band structure from vasprun file
    """
    from amset.plot.electronic_structure import ElectronicStructurePlotter

    plotter_kwargs = {
        "print_log": True,
        "interpolation_factor": kwargs["interpolation_factor"],
        "symprec": kwargs["symprec"],
        "energy_cutoff": kwargs["energy_cutoff"],
    }
    if is_vasprun_file(filename):
        plotter = ElectronicStructurePlotter.from_vasprun(filename, **plotter_kwargs)
    else:
        click.Abort("Unrecognised filetype, expecting a vasprun.xml file.")

    kpath = get_kpath(
        plotter.structure,
        mode=kwargs["kpath"],
        symprec=kwargs["symprec"],
        kpt_list=kwargs["kpoints"],
        labels=kwargs["labels"],
    )

    dos_kpoints = _get_dos_kpoints(plotter.structure, kwargs["dos_kpoints"])

    plt = plotter.get_plot(
        plot_band_structure=True,
        plot_dos=kwargs["plot_dos"],
        line_density=kwargs["line_density"],
        dos_kpoints=dos_kpoints,
        dos_estep=kwargs["dos_estep"],
        dos_aspect=kwargs["dos_aspect"],
        zero_to_efermi=kwargs["zero_to_efermi"],
        vbm_cbm_marker=kwargs["vbm_cbm_marker"],
        emin=kwargs["emin"],
        emax=kwargs["emax"],
        width=kwargs["width"],
        height=kwargs["height"],
        style=kwargs["style"],
        no_base_style=kwargs["no_base_style"],
        kpath=kpath,
    )

    save_plot(
        plt, "band", kwargs["directory"], kwargs["prefix"], kwargs["image_format"]
    )
    return plt


@plot.command()
@argument("filename", type=path_type)
@option("--ymin", default=None, type=float, help="minimum y-axis limit")
@option("--ymax", default=None, type=float, help="maximum y-axis limit")
@option("-d", "--doping-idx", metavar="N", help="doping index to plot")
@option("-t", "--temperature-idx", metavar="N", help="temperature index to plot")
@option(
    "-s",
    "--separate-rates",
    is_flag=True,
    default=False,
    help="whether to separate scattering mechanisms",
)
@option("-p", "--prefix", help="output filename prefix")
@option("--directory", type=path_type, help="file output directory")
@option("--image-format", default="pdf", type=image_type, help="image format")
@option("--style", help="path to matplotlib style specification")
@option("--no-base-style", default=False, is_flag=True, help="don't apply base style")
def rates(filename, **kwargs):
    """
    Plot scattering rates
    """
    from amset.plot.rates import RatesPlotter

    save_kwargs = [kwargs.pop(d) for d in ["directory", "prefix", "image_format"]]

    kwargs["doping_idx"] = _all_or_int(kwargs["doping_idx"])
    kwargs["temperature_idx"] = _all_or_int(kwargs["temperature_idx"])

    plotter = RatesPlotter(filename)
    plt = plotter.get_plot(**kwargs)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    save_plot(plt, "rates", *save_kwargs)
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


def is_vasprun_file(filename):
    return "xml" in filename


def get_kpath(structure, mode="pymatgen", symprec=_symprec, kpt_list=None, labels=None):
    r"""Get a Kpath object

    If a manual list of kpoints is supplied using the `kpt_list`
    variable, the `mode` option will be ignored.

    Args:
        structure: The structure.
        mode: Method used for calculating the
            high-symmetry path. The options are:

            pymatgen
                Use the paths from pymatgen.

            seekpath
                Use the paths from SeeK-path.

        symprec: The tolerance for determining the crystal symmetry.
        kpt_list: List of k-points to use, formatted as a list of subpaths, each
            containing a list of fractional k-points. For example:

            ```
                [ [[0., 0., 0.], [0., 0., 0.5]], [[0.5, 0., 0.], [0.5, 0.5, 0.]] ]
            ```

            will return points along `0 0 0 -> 0 0 1/2 | 1/2 0 0 -> 1/2 1/2 0`
        labels: The k-point labels. These should be provided as a list of strings for
            each subpath of the overall path. For example::

            ```
                [ ['Gamma', 'Z'], ['X', 'M'] ]
            ```

            combined with the above example for `kpt_list` would indicate the path:
            `Gamma -> Z | X -> M`. If no labels are provided, letters from A -> Z will
            be used instead.

    Returns:
        A Kpath object.
    """
    from sumo.symmetry import SeekpathKpath, PymatgenKpath, CustomKpath

    if kpt_list:
        kpath = CustomKpath(structure, kpt_list, labels, symprec=symprec)
    elif mode == "seekpath":
        kpath = SeekpathKpath(structure, symprec=symprec)
    else:
        kpath = PymatgenKpath(structure, symprec=symprec)

    return kpath


def _get_dos_kpoints(structure, kpoints):
    from amset.electronic_structure.kpoints import get_kpoint_mesh

    if kpoints is None:
        return None

    splits = kpoints.split()
    if len(splits) == 1:
        return get_kpoint_mesh(structure, float(splits[0]))
    else:
        return list(map(int, splits))
