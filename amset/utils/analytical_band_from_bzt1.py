# coding: utf-8
import matplotlib
import numpy as np
import warnings

matplotlib.use('agg')
from amset.utils.constants import Ry_to_eV
from pymatgen.core.units import Energy
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


__author__ = "Francesco Ricci and Alireza Faghaninia"
__copyright__ = "Copyright 2017, HackingMaterials"
__maintainer__ = "Francesco Ricci"

'''
The fitting algorythm is the Shankland-Koelling-Wood Fourier interpolation scheme,
implemented (for example) in the BolzTraP software package (boltztrap1).

Details of the interpolation method are available in:
(1) R.N. Euwema, D.J. Stukel, T.C. Collins, J.S. DeWitt, D.G. Shankland,
    Phys. Rev. 178 (1969)  1419–1423.
(2) D.D. Koelling, J.H. Wood, J. Comput. Phys. 67 (1986) 253–262.
(3) Madsen, G. K. & Singh, D. J. Computer Physics Communications 175, 67–71 (2006).


The coefficient for fitting are indeed calculated in BoltzTraP, not in this code.
Here, we just build the star functions using those coefficients.
Then, we also calculate the energy bands for each k-point in input.
'''


def get_energy(xkpt, engre, nwave, nsym, nstv, vec, vec2=None, out_vec2=None,
               br_dir=None, cbm=True, return_dde=True):
    '''
    Compute energy for a k-point from star functions
    Args:
        xkpt: k-point coordinates as array
        engre: matrix containing the coefficients of fitted band from get_engre() function
        nwave: number of G vectors
        nsym: number of symmetries
        nstv: number of vectors in a star function
        vec: the star vectors for each G vector and symmetry
        vec2: dot product of star vectors with cell matrix for each G vector
            and symmetry. Needed only to compute the derivatives of energy
        br_dir: cell matrix. Needed only to compute the derivatives of energy
        cbm: True if the considered band is a conduction band. False if it is a valence band
        out_vec2: outer product of vec2 with itself. It is calculated outside to improve performances
        return_dde (bool): if true, it also returns the second derivative of
            energy used to calculate the effective mass for example.

    Returns:
        ene: the electronic energy at the k-point in input
        dene: 1st derivative of the electronic energy at the k-point in input
        ddene: 2nd derivative of the electronic energy at the k-point in input
    '''
    sign = -1 if cbm == False else 1
    arg = 2 * np.pi * vec.dot(xkpt)
    tempc = np.cos(arg)
    spwre = (np.sum(tempc, axis=1) - (nsym - nstv))/nstv

    if br_dir is not None:
        temps = np.sin(arg)
        # np.newaxis adds a new dimensions so that the shape of temperatures (nwave,2)
        # converts to (nwave,2,1) so it can be projected to vec2 (nwave, 2, 3)
        dspwre = np.sum(vec2 * temps[:, :, np.newaxis], axis=1)
        dspwre /= nstv[:, np.newaxis]
        if return_dde:
            out_tempc = out_vec2 * (-tempc[:, :, np.newaxis, np.newaxis])
            ddspwre = np.sum(out_tempc, axis=1) / nstv[:, np.newaxis, np.newaxis]

    ene = spwre.dot(engre)
    if br_dir is not None:
        dene = np.dot(dspwre.T, engre)
        if return_dde:
            ddene = np.dot(ddspwre.T, engre)
            return sign * ene, dene, ddene
        else:
            return sign * ene, dene
    else:
        return sign * ene


def get_dos(energies,weights,e_min=None,e_max=None,e_points=None,width=0.2):
    """
    Args:
        energies: list of values in eV
                  from a previous interpolation over all the bands
                  and all the irreducible k-points
        weights:  list of multeplicities of each energies
        e_min:    starting energy (eV) of DOS
        e_max:    ending energy (eV) of DOS
        e_points: number of points of the get_dos
        width:    width in eV of the gaussians generated for each energy
        Returns:
        e_mesh:   energies in eV od the DOS
        dos:      density of states for each energy in e_mesh
    """
    if not e_min:
        e_min = min(energies)
    if not e_max:
        e_max = max(energies)

    height = 1.0 / (width * np.sqrt(2 * np.pi))
    if e_points:
        e_mesh, step = np.linspace(e_min, e_max,num=e_points, endpoint=True, retstep=True)
    else:
        e_mesh = np.array([en for en in energies])

    e_range = len(e_mesh)

    dos = np.zeros(e_range)
    for ene,w in zip(energies,weights):
        g = height * np.exp(-((e_mesh - ene) / width) ** 2 / 2.)
        dos += w * g
    return e_mesh,dos


class Analytical_bands(object):
    """This class is meant to read the BoltzTraP fitted band structure coefficients and facilitate calculation of
    energy and its first and second derivatives w.r.t wave vector, k."""
    def __init__(self, coeff_file):
        self.coeff_file = coeff_file


    def stern(self,g,nsym,symop):
        """
        Compute star function for a specific g vector
        Args:
            g: G vector in real space
            nsym: number of symmetries
            symop: matrixes of the symmetry operations

        Returns:
            nst: number of vectors in the star function calculated for the G vector
            stg: star vectors

        """

        trial = symop[:nsym].dot(g)
        stg = np.unique(trial.view(np.dtype((np.void, trial.dtype.itemsize*trial.shape[1])))).view(trial.dtype).reshape(-1, trial.shape[1])
        nst = len(stg)
        stg = np.concatenate((stg,np.zeros((nsym-nst,3))))
        return nst, stg



    def get_star_functions(self, latt_points, nsym, symop, nwave, br_dir=None):
        """
        Compute star functions for all G vectors and symmetries.
        Args:
            latt_points: all the G vectors
            nsym: number of symmetries
            symop: matrixes of the symmetry operations
            nwave: number of G vectors
            br_dir: cell matrix. Needed only to compute the derivatives of energy

        Returns:
            nstv: number of vectors in a star function for each G vector
            vec: the star funcions for each G vector and symmetry
            vec2: dot product of star vectors with cell matrix for each G vector
                    and symmetry. Needed only to compute the derivatives of energy
        """

        nstv = np.zeros(nwave,dtype='int')
        vec = np.zeros((nwave,nsym,3))
        if br_dir is not None:
            vec2 = np.zeros((nwave,nsym,3))

        for nw in range(nwave):
            nstv[nw], vec[nw]  = self.stern(latt_points[nw],nsym,symop)
            if br_dir is not None:
                vec2[nw] = vec[nw].dot(br_dir)

        #print vec
        if br_dir is not None:
            return nstv, vec, vec2
        else:
            return nstv, vec


    def get_engre(self,iband=None, return_bmin=False):
        """
        Get coefficients of interpolation from a custom output file from BoltzTraP.
        Some other info are also read and provided as output.

        Args:
            iband: list of indexes of the bands to fit (starting from 1). If nothing is passed
                    nothing is done, but the indexes of all the available bands in the file is printed.
                    The same happens if an index of a band not included is given.
                    if "A" is given, all the coefficients available are extracted
            return_bmin: True if the index of the first band contained in the file
                        is returned as output. False if it is not needed
        Returns:
            engre: matrix containing the coefficients of all the bands to fit
            latt_points: all the G vectors
            nwave: number of G vectors
            nsym: number of symmetries
            nsymop: max number of symmetries (usually 192, not used)
            symop: matrixes of the symmetry operations
            br_dir: cell matrix. Needed only to compute the derivatives of energy
            bmin: the index of the first band contained in the file
        """
        filename = self.coeff_file
        with open(filename) as f:
            egap, nwave, nsym, nsymop=f.readline().split()
            egap, nwave, nsym, nsymop=float(egap), int(nwave), int(nsym), int(nsymop)
            br_dir = np.fromstring(f.readline(),sep=' ',count=3*3).reshape((3,3)).astype('float')
            symop=np.fromstring(f.readline(),sep=' ',count=3*3*192).reshape((192,3,3)).astype('int')
            symop=np.transpose(symop,axes=(0,2,1))
            latt_points=np.zeros((nwave,3))
            engre=[]
            for i,l in enumerate(f):
                if i<nwave:
                    latt_points[i]=np.fromstring(l,sep=' ')
                elif i == nwave:
                    bmin, bmax = np.fromstring(l, sep=' ', dtype=int)
                    if iband == 'A':
                        iband = range(bmin,bmax+1)
                    elif iband == None:
                        print("Bands range: {}-{}".format(bmin,bmax))
                        return
                    elif len([ib for ib in iband if ib > bmax or ib < bmin]) > 0:
                        raise ValueError("at least one band is not in range : {}-{}".format(bmin,bmax))
                    iband2 = [nwave+(b-bmin+1) for b in iband]
                elif i in iband2:
                    engre.append(np.fromstring(l,sep=' '))
                    iband2.pop(iband2.index(i))
                    if len(iband2) == 0:
                        break
        if return_bmin:
            return engre, latt_points, nwave, nsym, nsymop, symop, br_dir, bmin
        else:
            return engre, latt_points, nwave, nsym, nsymop, symop, br_dir

    def get_extreme(self, kpt,iband,only_energy=False,cbm=True):
        engre,latt_points,nwave, nsym, nsymop, symop, br_dir = self.get_engre(iband)
        if only_energy == False:
            energy, denergy, ddenergy = get_energy(kpt,engre,nwave, nsym, nstv, vec, vec2, br_dir,cbm=cbm)
            return Energy(energy,"Ry").to("eV"), denergy, ddenergy
        else:
            energy = get_energy(kpt,engre,nwave, nsym, nstv, vec, cbm=cbm)
            return Energy(energy,"Ry").to("eV") # This is in eV automatically



    def get_dos_from_scratch(self,st,mesh,e_min,e_max,e_points,width=0.2, scissor=0.0, vbmidx = None):
        """
        Args:
        st:       pmg object of crystal structure to calculate symmetries
        mesh:     list of integers defining the k-mesh on which the dos is required
        e_min:    starting energy (eV) of dos
        e_max:    ending energy (eV) of dos
        e_points: number of points of the get_dos
        width:    width in eV of the gaussians generated for each energy
        Returns:
        e_mesh:   energies in eV od the DOS
        dos:      density of states for each energy in e_mesh
        """
        height = 1.0 / (width * np.sqrt(2 * np.pi))
        e_mesh, step = np.linspace(e_min, e_max,num=e_points, endpoint=True, retstep=True)
        e_range = len(e_mesh)
        cbm_new_idx = None

        if vbmidx:
            engre, latt_points, nwave, nsym, nsymop, symop, br_dir, bmin = self.get_engre(iband="A", return_bmin=True)
            cbm_new_idx = vbmidx - bmin + 1 # because now 0 is bmin and not
        else:
            engre, latt_points, nwave, nsym, nsymop, symop, br_dir = self.get_engre(iband="A")
            warnings.warn("The index of VBM / CBM is unknown; scissor is set to 0.0")
            scissor = 0.0

        nstv, vec = self.get_star_functions(latt_points,nsym,symop,nwave)
        ir_kpts = SpacegroupAnalyzer(st).get_ir_reciprocal_mesh(mesh)
        ir_kpts = [k[0] for k in ir_kpts]
        weights = [k[1] for k in ir_kpts]
        w_sum = float(sum(weights))
        weights = [w/w_sum for w in weights]
        dos = np.zeros(e_range)
        for kpt,w in zip(ir_kpts,weights):
            for b in range(len(engre)):
                energy = get_energy(kpt,engre[b], nwave, nsym, nstv, vec)*Ry_to_eV
                if b >= cbm_new_idx:
                    energy += scissor/2.
                else:
                    energy -= scissor/2.
                g = height * np.exp(-((e_mesh - energy) / width) ** 2 / 2.)
                dos += w * g
        if vbmidx:
            return e_mesh, dos, len(engre), bmin
        else:
            return e_mesh,dos, len(engre)

    def get_dos_standard(self,energies,weights,e_min,e_max,e_points,width=0.2):
        """
        Args:
        energies: matrix (num_kpoints,num_bands) of values in eV 
                  from a previous interpolation over all the bands (num_bands)
                  and all the irreducible k-points (num_kpoints)
        weights:  list of multeplicities of irreducible k-points
        e_min:    starting energy (eV) of DOS
        e_max:    ending energy (eV) of DOS
        e_points: number of points of the get_dos
        width:    width in eV of the gaussians generated for each energy
        Returns:
        e_mesh:   energies in eV od the DOS
        dos:      density of states for each energy in e_mesh
        """
        height = 1.0 / (width * np.sqrt(2 * np.pi))
        e_mesh, step = np.linspace(e_min, e_max,num=e_points, endpoint=True, retstep=True)
        e_range = len(e_mesh)

        dos = np.zeros(e_range)
        for kpt_ene,w in zip(energies,weights):
            for ene in kpt_ene:
                g = height * np.exp(-((e_mesh - ene) / width) ** 2 / 2.)
                dos += w * g
        return e_mesh,dos
