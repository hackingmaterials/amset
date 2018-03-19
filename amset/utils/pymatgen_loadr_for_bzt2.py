# spin polarized case not implemented yet
#
# example of usage:
# 
# data = Pymatgen_loader(vasprun_file)
#
# then use it as the object you get with:
#
# data = BTP.DFTData(vasp_dir)

from pymatgen.io.vasp import Vasprun
from pymatgen.io.ase import AseAtomsAdaptor


class Pymatgen_loader:
    def __init__(self, vasprun_file):
        vrun = Vasprun(vasprun_file)
        self.kpoints = np.array(vrun.actual_kpoints)
        self.structure = vrun.final_structure
        self.atoms = AseAtomsAdaptor.get_atoms(self.structure)
        if len(vrun.eigenvalues) == 1:
            e = list(vrun.eigenvalues.values())[0]
            self.ebands = e[:,:,0].transpose() * units.eV
            self.dosweight = 2.0
        elif len(vrun.eigenvalues) == 2:
            raise BaseException("spin bs case not implemented")
        
        self.lattvec = self.structure.lattice.matrix * units.Angstrom
        self.mommat = None
        self.fermi = vrun.efermi * units.eV
        self.nelect = vrun.parameters['NELECT']
        self.UCvol = self.structure.volume * units.Angstrom**3
        
    def get_lattvec(self):
        try:
            self.lattvec
        except AttributeError:
            self.lattvec = self.atoms.get_cell().T * Angstrom
        return self.lattvec
    
    def bandana(self, emin=-np.inf, emax=np.inf):
        bandmin = np.min(self.ebands, axis=1)
        bandmax = np.max(self.ebands, axis=1)
        II = np.nonzero(bandmin < emax)
        nemax = II[0][-1]
        II = np.nonzero(bandmax > emin)
        nemin = II[0][0]
        #BoltzTraP2.misc.info("BANDANA output")
        #for iband in range(len(self.ebands)):
            #BoltzTraP2.misc.info(iband, bandmin[iband], bandmax[iband], (
                #(bandmin[iband] < emax) & (bandmax[iband] > emin)))
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
