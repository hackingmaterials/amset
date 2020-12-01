import click
from click import argument, option

from amset.tools.common import path_type, zero_weighted_type
from amset.util import parse_doping, parse_temperatures

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

temp_doping_defaults = {"temperatures": [300], "doping": [-1e15, 1e15]}


@click.command()
@argument("filename", type=path_type)
@option(
    "-i",
    "--interpolation-factor",
    type=float,
    help="band structure interpolation factor",
)
@option(
    "-d",
    "--doping",
    metavar="D",
    type=parse_doping,
    help='doping concentrations (e.g. "1E13,1E14" or "1E13:1E20:8")',
)
@option(
    "-t",
    "--temperatures",
    metavar="T",
    type=parse_temperatures,
    help='temperatures (e.g. "300,400" or "300:1000:8")',
)
@option("--scissor", type=float, help="amount to scissor the band gap")
@option("--bandgap", type=float, help="manually set the band gap")
@option("--average/--no-average", default=False, help="report the harmonic average")
@option("--print-log/--no-log", default=True, help="whether to print log messages")
@option(
    "-z",
    "--zero-weighted-kpoints",
    help="how to process zero-weighted k-points",
    type=zero_weighted_type,
)
def eff_mass(filename, **kwargs):
    """
    Calculate conductivity effective mass.
    """
    import numpy as np
    from scipy import constants
    from tabulate import tabulate

    from amset.constants import bohr_to_cm, defaults
    from amset.core.run import Runner

    settings = {
        "scattering_type": ["CRT"],
        "constant_relaxation_time": 1e-14,
        "use_projections": True,
        "calculate_mobility": False,
        "separate_mobility": False,
        "write_log": False,
        "file_format": None,
    }
    for setting in defaults:
        if setting in kwargs and kwargs[setting] is not None:
            settings[setting] = kwargs[setting]
        elif setting == "doping" or setting == "temperatures":
            settings[setting] = temp_doping_defaults[setting]

    runner = Runner.from_vasprun(filename, settings)
    amset_data = runner.run()
    inv_cond = np.linalg.inv(amset_data.conductivity)
    doping_scale = 1 / bohr_to_cm ** 3
    doping = (np.abs(amset_data.doping) * doping_scale)[:, None, None, None]
    crt = settings["constant_relaxation_time"]

    masses = inv_cond * crt * doping * 10 ** 6 * constants.e ** 2 / constants.m_e
    mass_info = []
    for n, t in np.ndindex(amset_data.fermi_levels.shape):
        info = [amset_data.doping[n] * doping_scale, amset_data.temperatures[t]]
        if kwargs["average"]:
            eigs = np.linalg.eigvals(masses).real
            info.append((3 / (1 / eigs[n, t]).sum()))
        else:
            xyz = np.diagonal(masses, axis1=-2, axis2=-1)
            info.extend(xyz[n, t].tolist())
        mass_info.append(info)

    headers = ["conc [cm⁻³]", "temp [K]"]
    floatfmt = [".2e", ".1f"]

    if kwargs["average"]:
        headers.append("m*")
        floatfmt.append(".3f")
    else:
        headers.extend(["mₓₓ*", "mᵧᵧ*", "m₂₂*"])
        floatfmt.extend([".3f"] * 3)

    table = tabulate(
        mass_info,
        headers=headers,
        numalign="right",
        stralign="center",
        floatfmt=floatfmt,
    )

    click.echo("\nEffective masses:\n")
    click.echo(table)

    return mass_info
