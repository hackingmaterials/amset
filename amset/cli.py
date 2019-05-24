"""
This module contains a script for using amset from the command line.
"""

import argparse
import logging
import sys
import warnings
from typing import Optional

from pymatgen.core.structure import Structure
from amset import __version__

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__date__ = "December 17, 2018"


def amset(structure: Structure,
                         condenser_kwargs: Optional[dict] = None,
                         describer_kwargs: Optional[dict] = None,
                         ) -> str:
    """Gets the robocrystallographer description of a structure.

    Args:
        structure: A structure.
        condenser_kwargs: Keyword arguments that will be passed to
            :obj:`robocrys.condense.StructureCondenser`.
        describer_kwargs: Keyword arguments that will be passed to
            :obj:`robocrys.describe.StructureDescriber`.

    Returns:
        The description.
    """
    condenser_kwargs = condenser_kwargs if condenser_kwargs else {}
    describer_kwargs = describer_kwargs if describer_kwargs else {}

    sc = StructureCondenser(**condenser_kwargs)
    describer = StructureDescriber(**describer_kwargs)

    try:
        logging.info("Adding oxidation states...")
        structure.add_oxidation_state_by_guess(max_sites=-80)
    except ValueError:
        logging.warning("Could not add oxidation states!")

    condensed_structure = sc.condense_structure(structure)
    description = describer.describe(condensed_structure)
    logging.info(description)
    return description


def _get_parser():
    parser = argparse.ArgumentParser(
        description="amset is a tool to calculate carrier transport properties "
                    "from ab initio calculations",
        epilog="Author: {}, Version: {}, Last updated: {}".format(
            __author__, __version__, __date__))

    parser.add_argument('filename',
                        help="structure file or mpid")
    parser.add_argument('-c', '--conventional',
                        dest="use_conventional_cell",
                        action='store_true',
                        help="use the convention cell")
    parser.add_argument('-s', '--symmetry',
                        action='store_true',
                        dest="use_symmetry",
                        help="use symmetry to determine inequivalent sites")
    parser.add_argument('--symprec',
                        default=0.01,
                        help="symmetry tolerance")
    parser.add_argument('--no-simplify',
                        action='store_false',
                        dest="simplify_molecules",
                        help="don't simplify molecules when mineral matching")
    parser.add_argument('--no-iupac',
                        action="store_false",
                        dest="use_iupac_formula",
                        help="don't use IUPAC formula ordering")
    parser.add_argument('--no-common-formulas',
                        dest="use_common_formulas",
                        action="store_false",
                        help="don't use common formulas")
    parser.add_argument('--no-mineral',
                        dest="describe_mineral",
                        action="store_false",
                        help="don't describe the mineral information")
    parser.add_argument('--no-makeup',
                        dest="describe_component_makeup",
                        action="store_false",
                        help="don't describe the component makeup")
    parser.add_argument('--no-components',
                        dest="describe_components",
                        action="store_false",
                        help="don't describe the components")
    parser.add_argument('--no-symmetry-labels',
                        dest="describe_symmetry_labels",
                        action="store_false",
                        help="don't describe symmetry labels")
    parser.add_argument('--no-oxi',
                        dest="describe_oxidation_states",
                        action="store_false",
                        help="don't describe oxidation states")
    parser.add_argument('--no-bond',
                        dest="describe_bond_lengths",
                        action="store_false",
                        help="don't describe bond lengths")
    parser.add_argument('--precision',
                        metavar="P",
                        dest="bond_length_decimal_places",
                        default=2,
                        help="decimal places for bond lengths")
    parser.add_argument('--distorted-tol',
                        metavar="T",
                        dest="distorted_tol",
                        default=0.6,
                        help="order parameter below which sites are distorted")
    parser.add_argument('--anion-polyhedra',
                        dest="only_describe_cation_polyhedra_connectivity",
                        action="store_true",
                        help="describe anion polyhedra connectivity")
    parser.add_argument('--verbose-bonds',
                        dest="only_describe_bonds_once",
                        action="store_false",
                        help="describe bond lengths for each site")
    parser.add_argument('--latexify',
                        action="store_true",
                        help="format the description for use in LaTeX")
    return parser


def main():
    args = _get_parser().parse_args()
    args_dict = vars(args)

    condenser_keys = ['use_conventional_cell', "use_symmetry", "symprec",
                      "use_iupac_formula", "use_common_formulas"]
    describer_keys = ['describe_mineral', "describe_component_makeup",
                      "describe_components", "describe_symmetry_labels",
                      "describe_oxidation_states", "describe_bond_lengths",
                      "bond_length_decimal_places", "distorted_tol",
                      "only_describe_cation_polyhedra_connectivity",
                      "only_describe_bonds_once", "latexify"]

    condenser_kwargs = {key: args_dict[key] for key in condenser_keys}
    describer_kwargs = {key: args_dict[key] for key in describer_keys}

    logging.basicConfig(filename='robocrys.log', level=logging.INFO,
                        filemode='w', format='%(message)s')
    console = logging.StreamHandler()
    logging.info(" ".join(sys.argv[:]))
    logging.getLogger('').addHandler(console)

    warnings.filterwarnings("ignore", category=UserWarning,
                            module="pymatgen")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    try:
        structure = Structure.from_file(args.filename)

    except FileNotFoundError:
        from pymatgen.ext.matproj import MPRester

        mpr = MPRester()

        try:
            structure = mpr.get_entry_by_material_id(
                args.filename, inc_structure='final').structure
        except IndexError:
            logging.error("filename or mp-id not found.")
            sys.exit()

    robocrystallographer(structure, condenser_kwargs=condenser_kwargs,
                         describer_kwargs=describer_kwargs)
