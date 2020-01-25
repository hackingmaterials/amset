"""
This module contains a script for using amset from the command line.
"""

import click
from amset.tools.plot import plot
from amset.tools.run import run

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__date__ = "June 21, 2019"


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def cli():
    """
    AMSET is a tool to calculate carrier transport properties from ab initio
    calculation data
    """
    pass


cli.add_command(plot)
cli.add_command(run)
