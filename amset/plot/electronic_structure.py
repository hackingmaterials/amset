from amset.constants import amset_defaults as defaults
from amset.electronic_structure.interpolate import Interpolater
from pymatgen.electronic_structure.bandstructure import BandStructure


class ElectronicStructurePlotter(object):
    def __init__(
        self,
        bandstructure: BandStructure,
        nelect: int,
        interpolation_factor=5,
        print_log=defaults["print_log"],
    ):
        self.interpolation_factor = interpolation_factor
        self.print_log = print_log
        self.interpolater = Interpolater(bandstructure, nelect)

    def get_plot(
        self,
        zero_to_efermi=True,
        kpath=None,
        line_density=100,
        height=6,
        width=6,
        ymin=None,
        ymax=None,
        ylabel="Energy (eV)",
        plt=None,
        aspect=None,
        distance_factor=10,
        style=None,
        no_base_style=False,
    ):
        pass
