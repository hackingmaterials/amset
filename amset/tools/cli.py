"""
This module contains a script for using amset from the command line.
"""
import warnings

import click
from ruamel.yaml.error import MantissaNoDotYAML1_1Warning

from amset.tools.effmass import eff_mass
from amset.tools.phonon_frequency import phonon_frequency
from amset.tools.plot import plot
from amset.tools.run import run
from amset.tools.wavefunction import dump_wavefunction

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="amset")
warnings.filterwarnings("ignore", category=FutureWarning, module="scipy")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", MantissaNoDotYAML1_1Warning)


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def cli():
    """
    AMSET is a tool to calculate carrier transport properties from ab initio
    calculation data
    """
    pass


cli.add_command(plot)
cli.add_command(run)
cli.add_command(phonon_frequency)
cli.add_command(dump_wavefunction)
cli.add_command(eff_mass)
