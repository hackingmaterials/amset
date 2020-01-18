"""
This module contains a script for using amset from the command line.
"""

import click
import warnings

from collections import defaultdict
from typing import Dict, Any

from click import option
from amset.log import initialize_amset_logger
from amset.run import AmsetRunner
from amset.constants import amset_defaults
from amset.util import parse_deformation_potential, parse_temperatures, parse_doping

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__date__ = "June 21, 2019"


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
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
@option("--acceptor-charge", type=float, help="acceptor defect charge")
@option("--donor-charge", type=float, help="donor defect charge")
@option("--pop-frequency", type=float, help="polar optical phonon frequency [THz]")
@option(
    "--energy-cutoff", type=float, help="energy cut-off for band interpolation [eV]"
)
@option(
    "--fd-tol",
    type=float,
    help="Fermi-Dirac tolerance below which scattering rates are not "
    "calculated [%%]",
)
@option("--dos-estep", type=float, help="dos energy step [eV]")
@option("--symprec", type=float, help="symmetry precision")
@option("--nworkers", type=float, help="number of processors to use")
@option(
    "--calculate-mobility/--no-calculate-mobility",
    default=True,
    help="whether to calculate mobility",
)
@option(
    "--separate-mobility/--no-separate-mobility",
    default=True,
    help="whether to separate the individual scattering rate mobilities",
)
@option("--file-format", help="output file format [options: json, yaml, txt, dat]")
@option(
    "--write-input/--no-write-input", default=False, help="write input settings to file"
)
@option(
    "--write-mesh/--no-write-mesh",
    default=False,
    help="write mesh data, including band energies and scattering rates",
)
@option("--print-log/--no-log", default=True, help="whether to print log messages")
def main(**kwargs):
    """
    AMSET is a tool to calculate carrier transport properties from ab initio
    calculation data
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="amset")
    warnings.filterwarnings("ignore", category=FutureWarning, module="scipy")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    if kwargs["print_log"] is not False:
        initialize_amset_logger()

    settings_override: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for setting in amset_defaults:
        if setting in kwargs and kwargs[setting] is not None:
            settings_override[setting] = kwargs[setting]

    runner = AmsetRunner.from_directory(
        directory=kwargs["directory"],
        vasprun=kwargs["vasprun"],
        settings_file=kwargs["settings"],
        settings_override=settings_override,
    )

    runner.run()
