"""
This module contains a script for using amset from the command line.
"""

from collections import defaultdict
from typing import Any, Dict

import click
from click import option

from amset.tools.common import zero_weighted_type
from amset.util import parse_deformation_potential, parse_doping, parse_temperatures

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


@click.command()
@option(
    "--directory", default=".", help="path to directory with vasprun and settings files"
)
@option("-v", "--vasprun", help="path to vasprun.xml file")
@option("-s", "--settings", help="path to settings file")
@option(
    "-i",
    "--interpolation-factor",
    type=float,
    help="band structure interpolation factor",
)
@option(
    "--d",
    "--doping",
    metavar="D",
    type=parse_doping,
    help='doping concentrations (e.g. "1E13,1E14" or "1E13:1E20:8")',
)
@option(
    "--t",
    "--temperatures",
    metavar="T",
    type=parse_temperatures,
    help='temperatures (e.g. "300,400" or "300:1000:8")',
)
@option("--scissor", type=float, help="amount to scissor the band gap")
@option("--bandgap", type=float, help="manually set the band gap")
@option(
    "-z",
    "--zero-weighted-kpoints",
    help="how to process zero-weighted k-points",
    type=zero_weighted_type,
)
@option(
    "--free-carrier-screening/--no-free-carrier-screening",
    default=None,
    help="whether free carriers will screen POP and PIE rates [default: False]",
)
@option("--scattering-type", type=float, help="scattering mechanisms to include")
@option(
    "--high-frequency-dielectric", type=float, help="high-frequency dielectric constant"
)
@option("--static-dielectric", type=float, help="static dielectric constant")
@option("--elastic-constant", type=float, help="elastic constant [GPa]")
@option(
    "--deformation-potential",
    metavar="D",
    type=parse_deformation_potential,
    help='deformation potential [eV] (e.g. "7.4" or "7.4,6.8")',
)
@option("--piezoelectric-constant", type=float, help="piezoelectric constant")
@option("--defect-charge", type=float, help="defect charge")
@option("--compensation-factor", type=float, help="defect compensation factor")
@option("--pop-frequency", type=float, help="polar optical phonon frequency [THz]")
@option("--mean-free-path", type=float, help="set the mean free path of electrons [nm]")
@option("--constant-relaxation-time", type=float, help="constant relaxation time [s]")
@option(
    "--energy-cutoff",
    type=float,
    help="energy cut-off for band electronic_structure [eV]",
)
@option(
    "--fd-tol",
    type=float,
    help="Fermi-Dirac tolerance below which scattering rates are not "
    "calculated [%%]",
)
@option(
    "--cache-wavefunction/--no-cache-wavefunction",
    default=None,
    help="cache wavefunction coefficients; beware increased memory usage [default: True]",
)
@option("--dos-estep", type=float, help="dos energy step [eV]")
@option("--symprec", type=float, help="symmetry precision")
@option("--nworkers", type=int, help="number of processors to use")
@option(
    "--calculate-mobility/--no-calculate-mobility",
    default=None,
    help="whether to calculate mobility",
)
@option(
    "--separate-mobility/--no-separate-mobility",
    default=None,
    help="whether to separate the individual scattering rate mobilities",
)
@option("--file-format", help="output file format [options: json, yaml, txt, dat]")
@option(
    "--write-input/--no-write-input", default=None, help="write input settings to file"
)
@option(
    "--write-mesh/--no-write-mesh",
    default=None,
    help="write mesh data, including band energies and scattering rates",
)
@option("--print-log/--no-log", default=True, help="whether to print log messages")
def run(**kwargs):
    """
    Run AMSET on an ab initio band structure
    """

    from amset.constants import defaults
    from amset.core.run import Runner
    from amset.log import initialize_amset_logger

    if kwargs["print_log"] is not False:
        initialize_amset_logger()

    settings_override: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for setting in defaults:
        if setting in kwargs and kwargs[setting] is not None:
            settings_override[setting] = kwargs[setting]

    runner = Runner.from_directory(
        directory=kwargs["directory"],
        input_file=kwargs["vasprun"],
        settings_file=kwargs["settings"],
        settings_override=settings_override,
    )

    runner.run()
