import os

from BoltzTraP2 import sphere, fite
from amset.logging import LoggableMixin
from amset.utils.analytical_band_from_bzt1 import AnalyticalBands
from amset.utils.band_interpolation import get_energy_args
from pymatgen.electronic_structure.boltztrap import BoltztrapRunner
from pymatgen.electronic_structure.boltztrap2 import BandstructureLoader


class Interpolater(LoggableMixin):

    def __init__(self, method, band_structure, num_electrons, coeff_file=None,
                 max_temperature=None, calc_dir='.'):
        self.method = method
        self._band_structure = band_structure
        self._num_electrons = num_electrons
        self._coeff_file = coeff_file
        self._max_temperature = max_temperature
        self._calc_dir = calc_dir
        self._interp_params = None
        self._analytical_bands = None

    def initialize(self):
        if '1' in self.method and not self._coeff_file:
            self.logger.info('No coefficient file specified, running BoltzTraP '
                             'to generate it.')

            # do NOT set scissor in runner
            btr = BoltztrapRunner(
                bs=self._band_structure, nelec=self._num_electrons,
                run_type='BANDS', doping=[1e20], tgrid=300,
                tmax=max([self._max_temperature, 300]))
            dirpath = btr.run(path_dir=self._calc_dir)
            self._coeff_file = os.path.join(dirpath, 'fort.123')

            if not os.path.exists(self._coeff_file):
                self.log_raise(RuntimeError,
                               'Coefficient file was not generated properly. '
                               'This requires a modified version of BoltzTraP. '
                               'See the patch_for_boltztrap" folder for more '
                               'information')
            else:
                self.logger.info('Finished generating coefficient file. Set '
                                 'coeff_file variable to {} to skip this in the'
                                 'future'.format(self._coeff_file))

            self._analytical_bands = AnalyticalBands(coeff_file=self._coeff_file)

        elif '2' in self.method:
            bz2_data = BandstructureLoader(
                self._band_structure, structure=self._band_structure.structure,
                nelect=self._num_electrons)
            equivalences = sphere.get_equivalences(
                atoms=bz2_data.atoms, nkpt=len(bz2_data.kpoints) * 5,
                magmom=None)
            lattvec = bz2_data.get_lattvec()
            coeffs = fite.fitde3D(bz2_data, equivalences)
            self._interp_params = (equivalences, lattvec, coeffs)

    def get_band_energies(self, kpoints, band_indices):
        """

        :param kpoints:
        :param band_indices:
        :return:
        """
        if '1' in self.method:
            self.logger.debug("Interpolating bands from coefficient file")
            interpolation_params = get_energy_args(
                self._analytical_bands, band_indices)
        elif '2' in self.method:

        self.logger.debug("band_indices: {}".format(band_indices))






