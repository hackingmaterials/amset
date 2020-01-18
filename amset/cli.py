"""
This module contains a script for using amset from the command line.
"""

import argparse
import warnings

from collections import defaultdict
from typing import Dict, Any

from amset.log import initialize_amset_logger
from amset.run import AmsetRunner
from amset import __version__
from amset.constants import amset_defaults
from amset.util import (
    parse_deformation_potential,
    parse_temperatures,
    parse_doping,
)

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__date__ = "June 21, 2019"


def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="amset")
    warnings.filterwarnings("ignore", category=FutureWarning, module="scipy")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    args = _get_parser().parse_args()
    args_dict = vars(args)

    if args.print_log is not False:
        initialize_amset_logger()

    settings_override: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for section in amset_defaults:
        for setting in amset_defaults[section]:
            if setting in args_dict and args_dict[setting] is not None:
                settings_override[section][setting] = args_dict[setting]

    runner = AmsetRunner.from_directory(
        directory=args.directory,
        vasprun=args.vasprun,
        settings_file=args.settings,
        settings_override=settings_override,
    )

    runner.run()


def _get_parser():
    parser = argparse.ArgumentParser(
        description="AMSET is a tool to calculate carrier transport properties "
        "from ab initio calculation data",
        epilog="Author: {}, Version: {}, Last updated: {}".format(
            __author__, __version__, __date__
        ),
    )

    parser.add_argument(
        "--directory",
        metavar="D",
        default=".",
        help="path to directory with vasprun and settings files",
    )

    parser.add_argument(
        "-v", "--vasprun", metavar="V", default=None, help="path to vasprun.xml file"
    )

    parser.add_argument(
        "-s", "--settings", metavar="S", default=None, help="path to settings file"
    )

    parser.add_argument(
        "-i",
        "--interpolation-factor",
        metavar="F",
        default=None,
        type=float,
        help="band structure interpolation factor",
    )

    parser.add_argument(
        "--d",
        "--doping",
        metavar="D",
        default=None,
        type=parse_doping,
        help='doping concentrations (e.g. "1E13,1E14" or "1E13:1E20:8")',
    )

    parser.add_argument(
        "--t",
        "--temperatures",
        metavar="T",
        default=None,
        type=parse_temperatures,
        help='temperatures (e.g. "300,400" or "300:1000:8")',
    )

    parser.add_argument(
        "--scissor",
        metavar="S",
        default=None,
        type=float,
        help="amount to scissor the band gap",
    )

    parser.add_argument(
        "--bandgap",
        metavar="B",
        default=None,
        type=float,
        help="automatically set scissor to make the band gap this value",
    )

    parser.add_argument(
        "--scattering-type",
        metavar="M",
        default=None,
        type=float,
        help="scattering mechanisms to include [default: auto]",
    )

    parser.add_argument(
        "--high-frequency-dielectric",
        metavar="ε_∞",
        default=None,
        type=float,
        help="high-frequency dielectric constant",
    )

    parser.add_argument(
        "--static-dielectric",
        metavar="ε_s",
        default=None,
        type=float,
        help="static dielectric constant",
    )

    parser.add_argument(
        "--elastic-constant",
        metavar="C",
        default=None,
        type=float,
        help="elastic constant [GPa]",
    )

    parser.add_argument(
        "--deformation-potential",
        metavar="E_D",
        default=None,
        type=parse_deformation_potential,
        help='deformation potential [eV] (e.g. "7.4" or "7.4,6.8")',
    )

    parser.add_argument(
        "--piezoelectric-constant",
        metavar="P",
        default=None,
        type=float,
        help="piezoelectric constant",
    )

    parser.add_argument(
        "--acceptor-charge",
        metavar="C",
        default=None,
        type=float,
        help="acceptor defect charge",
    )

    parser.add_argument(
        "--donor-charge",
        metavar="C",
        default=None,
        type=float,
        help="donor defect charge",
    )

    parser.add_argument(
        "--pop-frequency",
        metavar="ω",
        default=None,
        type=float,
        help="polar optical phonon frequency [THz]",
    )

    parser.add_argument(
        "--energy-cutoff",
        metavar="E",
        default=None,
        type=float,
        help="energy cut-off for band interpolation [eV]",
    )

    parser.add_argument(
        "--fd-tol",
        metavar="T",
        default=None,
        type=float,
        help="Fermi-Dirac tolerance below which scattering rates are not "
        "calculated [%%]",
    )

    parser.add_argument(
        "--dos-estep",
        metavar="E",
        default=None,
        type=float,
        help="dos energy step [eV]",
    )

    parser.add_argument(
        "--symprec", metavar="S", default=None, type=float, help="symmetry precision"
    )

    parser.add_argument(
        "--nworkers",
        metavar="N",
        default=None,
        type=float,
        help="number of processors to use",
    )

    parser.add_argument(
        "--no-calculate-mobility",
        default=None,
        dest="calculate_mobility",
        action="store_false",
        help="don't calculate mobility",
    )

    parser.add_argument(
        "--no-separate-mobility",
        default=None,
        dest="separate_scattering_mobilities",
        action="store_false",
        help="don't separate the individual scattering rate mobilities",
    )

    parser.add_argument(
        "--file-format",
        metavar="F",
        default=None,
        help="output file format [options: json, yaml, txt, dat]",
    )

    parser.add_argument(
        "--write-input",
        default=None,
        action="store_true",
        help="write input settings to disk",
    )

    parser.add_argument(
        "--write-mesh",
        default=None,
        action="store_true",
        help="write mesh data, including band energies and scattering rates",
    )

    parser.add_argument(
        "--no-log",
        default=None,
        dest="print_log",
        action="store_false",
        help="don't print log messages",
    )

    return parser
