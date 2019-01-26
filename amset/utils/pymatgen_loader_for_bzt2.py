# spin polarized case not implemented yet
#
# example of usage:
# 
# data = Pymatgen_loader(vasprun_file)
#
# then use it as the object you get with:
#
# data = BTP.DFTData(vasp_dir)

import numpy as np

from pymatgen import Spin
from pymatgen.io.ase import AseAtomsAdaptor
from BoltzTraP2 import units

from pymatgen.io.vasp import Vasprun


class PymatgenLoader(object):

    def __init__(self, band_structure, num_electrons):
        self.kpoints = np.array([k.frac_coords for k in band_structure.kpoints])
        self.structure = band_structure.structure

        self.atoms = AseAtomsAdaptor.get_atoms(self.structure)

        if len(band_structure.bands) == 1:
            self.ebands = band_structure.bands[Spin.up] * units.eV
            self.dosweight = 2.0

        else:
            raise BaseException("Spin polarized band structure not supported")

        self.lattvec = self.atoms.get_cell().T * units.Angstrom
        self.mommat = None
        self.fermi = band_structure.efermi * units.eV
        self.nelect = num_electrons
        self.UCvol = self.structure.volume * units.Angstrom ** 3

    @staticmethod
    def from_vasprun(vasprun):
        if isinstance(vasprun, str):
            vasprun = Vasprun(vasprun)

        band_structure = vasprun.get_band_structure()
        num_electrons = vasprun.parameters['NELECT']

        return PymatgenLoader(band_structure, num_electrons)

    def get_lattvec(self):
        try:
            self.lattvec
        except AttributeError:
            self.lattvec = self.atoms.get_cell().T * units.Angstrom
        return self.lattvec

    def bandana(self, emin=-np.inf, emax=np.inf):
        bandmin = np.min(self.ebands, axis=1)
        bandmax = np.max(self.ebands, axis=1)
        ii = np.nonzero(bandmin < emax)
        nemax = ii[0][-1]
        ii = np.nonzero(bandmax > emin)
        nemin = ii[0][0]
        self.ebands = self.ebands[nemin:nemax]
        if self.mommat is not None:
            self.mommat = self.mommat[:, nemin:nemax, :]
        # Removing bands may change the number of valence electrons
        self.nelect -= self.dosweight * nemin
        return nemin, nemax

    def get_volume(self):
        try:
            self.UCvol
        except AttributeError:
            lattvec = self.get_lattvec()
            self.UCvol = np.abs(np.linalg.det(lattvec))
        return self.UCvol


class BandstructureLoader:
    def __init__(self, pmg_bs_obj, structure=None, nelect=None):
        self.kpoints = np.array([kp.frac_coords for kp in pmg_bs_obj.kpoints])

        if structure is None:
            try:
                self.structure = pmg_bs_obj.structure
            except:
                BaseException('No structure found in the band_structure obj.')

        self.atoms = AseAtomsAdaptor.get_atoms(self.structure)

        if len(pmg_bs_obj.bands) == 1:
            e = list(pmg_bs_obj.bands.values())[0]
            self.ebands = e * units.eV
            self.dosweight = 2.0
        elif len(pmg_bs_obj.bands) == 2:
            raise BaseException("spin band_structure case not implemented")

        self.lattvec = self.atoms.get_cell().T * units.Angstrom
        self.mommat = None
        self.fermi = pmg_bs_obj.efermi * units.eV

        self.nelect = nelect
        self.UCvol = self.structure.volume * units.Angstrom ** 3

    def get_lattvec(self):
        try:
            self.lattvec
        except AttributeError:
            self.lattvec = self.atoms.get_cell().T * units.Angstrom
        return self.lattvec

    def bandana(self, emin=-np.inf, emax=np.inf):
        bandmin = np.min(self.ebands, axis=1)
        bandmax = np.max(self.ebands, axis=1)
        ii = np.nonzero(bandmin < emax)
        nemax = ii[0][-1]
        ii = np.nonzero(bandmax > emin)
        nemin = ii[0][0]
        # BoltzTraP2.misc.info("BANDANA output")
        # for iband in range(len(self.ebands)):
        # BoltzTraP2.misc.info(iband, bandmin[iband], bandmax[iband], (
        # (bandmin[iband] < emax) & (bandmax[iband] > emin)))
        self.ebands = self.ebands[nemin:nemax]
        if self.mommat is not None:
            self.mommat = self.mommat[:, nemin:nemax, :]
        # Removing bands may change the number of valence electrons
        if self.nelect is not None:
            self.nelect -= self.dosweight * nemin
        return nemin, nemax

    def get_volume(self):
        try:
            self.UCvol
        except AttributeError:
            lattvec = self.get_lattvec()
            self.UCvol = np.abs(np.linalg.det(lattvec))
        return self.UCvol
