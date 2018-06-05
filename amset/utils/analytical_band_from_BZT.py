# coding: utf-8
import matplotlib
import numpy as np
import warnings

matplotlib.use('agg')
from matplotlib.pylab import plot, show, scatter
from math import pi
from pymatgen.core.units import Energy
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from amset.utils.constants import hbar, m_e, Ry_to_eV, A_to_m, A_to_nm, m_to_cm, e


__author__ = "Francesco Ricci and Alireza Faghaninia"
__copyright__ = "Copyright 2017, HackingMaterials"
__maintainer__ = "Francesco Ricci"

'''
The fitting algorythm is the Shankland-Koelling-Wood Fourier interpolation scheme,
implemented (for example) in the BolzTraP software.

Details of the interpolation method are available in:
(1) R.N. Euwema, D.J. Stukel, T.C. Collins, J.S. DeWitt, D.G. Shankland,
    Phys. Rev. 178 (1969)  1419–1423.
(2) D.D. Koelling, J.H. Wood, J. Comput. Phys. 67 (1986) 253–262.
(3) Madsen, G. K. & Singh, D. J. Computer Physics Communications 175, 67–71 (2006).


The coefficient for fitting are indeed calculated in BoltzTraP, not in this code.
Here, we just build the star functions using those coefficients.
Then, we also calculate the energy bands for each k-point in input.
'''

def norm(v):
    """method to quickly calculate the norm of a vector (v: 1x3 or 3x1) as
    numpy.linalg.norm is slower if only used for one vector"""
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


def outer(v1, v2):
    """returns the outer product of vectors v1 and v2. This is to be used
    instead of numpy.outer which is ~3x slower if only used for 2 vectors."""
    return np.array([[v1[0] * v2[0], v1[0] * v2[1], v1[0] * v2[2]],
                     [v1[1] * v2[0], v1[1] * v2[1], v1[1] * v2[2]],
                     [v1[2] * v2[0], v1[2] * v2[1], v1[2] * v2[2]]])


def get_poly_energy(kpt, poly_bands, type, ib=0, bandgap=1, all_values = False):
    """
    Args:
        kpt (list): coordinates of a given k-point in the actual cartesian coordinates and NOT fractional coordinates
        rotations: symmetry rotation operations
        translations: symmetry translational operations
        poly_bands [[lists]]: each member of the first list represents a band: in each band a list of lists of lists
            should contain a list of two-member lists: the two members are: the coordinates of extrema k-point and its
            symmetrically equivalent points and another two members list of
            [first member: energy offset from the main extremum (i.e. CBM/VBM), second member: the effective mass]
            example poly_bands = [[ [[[0.5, 0.5, 0.5]], [0, 0.1]], [[[0, 0, 0]], [0.5, 0.2]]]] represents a
            band structure with a single band; this parabolic band (hbar**2k**2/2m*) at point X
            where k is norm(k-[0.5,0.5,0.5]) if the k-point is closer to X than Gamma. Additionally, this band has
            another extremum at an energy level 0.5 eV above the first/main extremum/CBM. If type is "p" the band structure
            would be a mirror image. VBM is always set to 0.0 eV and the CBM is set to the bandgap

        type (str): "n" or "p"
        ib (int): the band index, 0 is for the first band and maximum allowed value is len(poly_bands)-1
        bandgap (float): the targetted band gap of the band structure
    Returns:
    """
    # The sign of energy from type; e.g. p-type energies are negative (VBM=0.0)
    kpt = np.array(kpt)
    sgn = (-1)**(["p", "n"].index(type)+1)
    band_shapes = poly_bands[ib]
    min_kdist = 1e32
    allE = []
    for ks, c in band_shapes:
        #ks: all symmetrically equivalent k-points to the extremum k in 1st BZ
        for k in ks:
            distance = norm(kpt-k)
            allE.append(bandgap * ["p", "n"].index(type) + sgn * (c[0] + \
                    hbar ** 2 * (distance) ** 2 / (2 * m_e * c[1]) * e * 1e18))
            if distance < min_kdist:
                min_kdist = distance
                coefficients = c
    eff_m = coefficients[1]
    energy = sgn * bandgap/2.0
    energy += sgn*(coefficients[0] + hbar**2 * min_kdist**2 / (2*m_e*eff_m) * e*1e18) # last part is unit conv. to eV
    v = hbar*min_kdist/(m_e*eff_m) *1e11*e #last part is unit conversion to cm/s
    if not all_values:
        return energy, np.array([v, v, v]), sgn*np.array(
            [[eff_m, 0.0, 0.0], [0.0, eff_m, 0.0], [0.0, 0.0, eff_m]])
    else:
        return allE, min_kdist



def get_energy(xkpt, engre, nwave, nsym, nstv, vec, vec2=None, out_vec2=None,
               br_dir=None, cbm=True):
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
        # np.newaxis adds a new dimensions so that the shape of temps (nwave,2)
        # converts to (nwave,2,1) so it can be projected to vec2 (nwave, 2, 3)
        dspwre = np.sum(vec2 * temps[:, :, np.newaxis], axis=1)
        dspwre /= nstv[:, np.newaxis]
        out_tempc = out_vec2 * (-tempc[:, :, np.newaxis, np.newaxis])
        ddspwre = np.sum(out_tempc, axis=1) / nstv[:, np.newaxis, np.newaxis]

    ene = spwre.dot(engre)
    if br_dir is not None:
        dene = np.dot(dspwre.T, engre)
        ddene = np.dot(ddspwre.T, engre)
        return sign * ene, dene, ddene
    else:
        return sign * ene


def get_dos_from_poly_bands(st, reclat_matrix, mesh, e_min, e_max, e_points, poly_bands, bandgap, width=0.1, SPB_DOS=False, all_values=False):
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
    ir_kpts_n_weights = SpacegroupAnalyzer(st).get_ir_reciprocal_mesh(mesh)
    ir_kpts = [k[0] for k in ir_kpts_n_weights]
    weights = [k[1] for k in ir_kpts_n_weights]

    ir_kpts = [reclat_matrix.get_cartesian_coords(k)/A_to_nm for k in ir_kpts]

    w_sum = float(sum(weights))
    dos = np.zeros(e_range)

    if SPB_DOS:
        volume = st.volume
        for band in poly_bands:
            for valley in band:
                offset = valley[-1][0] # each valley has a list of k-points (valley[0]) and [offset, m*] (valley[1]) info
                m_eff = valley[-1][1]
                degeneracy = len(valley[0])
                for ie, energy in enumerate(e_mesh):
                    dos_temp = volume/(2*pi**2)*(2*m_e*m_eff/hbar**2)**1.5 * 1e-30/e**1.5
                    if energy <= -bandgap/2.0-offset:
                        # dos_temp *= (-energy-offset)**0.5
                        dos_temp *= (-energy+bandgap/2.0+offset)**0.5
                    elif energy >=bandgap/2.0+offset:
                        # dos_temp *= (energy-bandgap-offset)**0.5
                        dos_temp *= (energy-bandgap/2.0-offset)**0.5
                    else:
                        dos_temp = 0
                    dos[ie] += dos_temp * degeneracy
    else:
        all_energies = []
        all_ks = []
        for kpt,w in zip(ir_kpts,weights):
            for tp in ["n", "p"]:
                for ib in range(len(poly_bands)):
                    if all_values:
                        energy_list, k_dist = get_poly_energy(kpt, poly_bands, tp, ib=ib, bandgap=bandgap,
                                                           all_values=all_values)
                        all_energies += energy_list
                        all_ks += [k_dist]*len(energy_list)
                        for energy in energy_list:
                            g = height * np.exp(-((e_mesh - energy) / width) ** 2 / 2.)
                            dos += w/w_sum * g
                    else:
                        energy, v, m_eff = get_poly_energy(kpt, poly_bands, tp, ib=ib, bandgap=bandgap, all_values=all_values)
                        g = height * np.exp(-((e_mesh - energy) / width) ** 2 / 2.)
                        dos += w/w_sum * g
        if all_values:
            scatter(all_ks, all_energies)
            show()
    return e_mesh,dos


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


if __name__ == "__main__":
    # user inputs
    # cbm_bidx = 15 # GaAs
    cbm_bidx = 10 # InP
    # kpts = np.array([[0.5, 0.5, 0.5]])
    kpts = np.array(
        [[-0.1, 0.19999999999999998, 0.1], [0.1, -0.19999999999999998, -0.1], [0.1, -0.1, -0.19999999999999998],
         [-0.1, 0.1, 0.19999999999999998], [-0.19999999999999998, 0.1, -0.1], [0.19999999999999998, -0.1, 0.1],
         [0.19999999999999998, 0.1, -0.1], [-0.19999999999999998, -0.1, 0.1], [0.1, -0.1, 0.19999999999999998],
         [-0.1, 0.1, -0.19999999999999998], [-0.1, -0.19999999999999998, 0.1], [0.1, 0.19999999999999998, -0.1],
         [0.09999999999999998, 0.3, 0.19999999999999998], [-0.09999999999999998, -0.3, -0.19999999999999998],
         [-0.09999999999999998, -0.19999999999999998, -0.3], [0.09999999999999998, 0.19999999999999998, 0.3],
         [0.19999999999999998, 0.09999999999999998, 0.3], [-0.19999999999999998, -0.09999999999999998, -0.3],
         [-0.3, -0.09999999999999998, -0.19999999999999998], [0.3, 0.09999999999999998, 0.19999999999999998],
         [0.2, 0.3, 0.1], [-0.2, -0.3, -0.1], [0.3, 0.19999999999999998, 0.09999999999999998],
         [-0.3, -0.19999999999999998, -0.09999999999999998]])

    # kpts = [[ 0.,  0.,  0.], [ 0.42105263,  0.42105263,  0.        ]] # Si kVBM and kCBM respectively
    kpts = [[ 0.,  0.,  0.], [0.15, 0.15, 0.15]]
    # coeff_file = '../test_files/PbTe/fort.123'
    # coeff_file = "../test_files/GaAs/fort.123_GaAs_1099kp"
    # coeff_file = "../../test_files/GaAs/fort.123_GaAs_1099kp"
    # coeff_file = "../test_files/Si/Si_fort.123"
    # coeff_file = "/Users/alirezafaghaninia/Documents/boltztrap_examples/SnSe2/boltztrap_vdw_better_geom_dense/boltztrap/fort.123"
    # coeff_file = "../../test_files/GaAs/nscf-uniform/boltztrap/fort.123"
    coeff_file="/Users/alirezafaghaninia/Documents/py3/py3_codes/thermoelectrics_work/thermoelectrics_work/amset_examples/InP_mp-20351/boltztrap/fort.123"

    analytical_bands = Analytical_bands(coeff_file=coeff_file)
    # read the coefficients file
    engre, latt_points, nwave, nsym, nsymop, symop, br_dir = analytical_bands.get_engre(iband=[cbm_bidx])
    #generate the star functions only one time
    nstv, vec, vec2 = analytical_bands.get_star_functions(latt_points,nsym,symop,nwave,br_dir=br_dir)
    out_vec2 = np.zeros((nwave,max(nstv),3,3))
    for nw in range(nwave):
        for i in range(nstv[nw]):
            out_vec2[nw,i]= outer(vec2[nw,i],vec2[nw,i])
            
    # setup
    # en, den, dden = [], [], []
    # for kpt in kpts:
    #     energy, de, dde = get_energy(kpt,engre[0], nwave, nsym, nstv, vec, vec2, out_vec2, br_dir)
    #     en.append(energy*Ry_to_eV)
    #     den.append(de)
    #     dden.append(dde*2*pi)

    # print("outputs:")
    # print("Energy: {}".format(en))
    # print("1st derivate:")
    # print(den)
    # print("2nd derivative:")
    # print(dde)
    # m_tensor = hbar ** 2 /(dde*4*pi**2) / m_e / A_to_m ** 2 * e * Ry_to_eV # m_tensor: the last part is unit conversion
    # print("effective mass tensor")
    # print(m_tensor)


    # print("group velocity:")
    # v = de /hbar*A_to_m*m_to_cm * Ry_to_eV # to get v in units of cm/s
    # print(v)


    # kpt = np.array([0.1, 0.2, 0.3])
    kpt = np.array([0.0, 0.0, 0.0])
    # vrun = Vasprun('vasprun.xml')
    vrun = Vasprun('/Users/alirezafaghaninia/Documents/py3/py3_codes/thermoelectrics_work/thermoelectrics_work/amset_examples/InP_mp-20351/vasprun.xml')
    _rec_lattice = vrun.final_structure.lattice.reciprocal_lattice
    MATRIX = _rec_lattice.matrix
    dir_matrix = vrun.final_structure.lattice.matrix
    st = vrun.final_structure
    energy, de, dde = get_energy(kpt, engre[0], nwave, nsym, nstv, vec, vec2,
                                 out_vec2, br_dir)
    velocity = abs(np.dot(MATRIX, de)) / hbar * A_to_m * m_to_cm * Ry_to_eV

    # def get_cartesian_coords(frac_k):
    #     return np.dot(MATRIX, frac_k)


    print('cartesian with old transformation')
    print(_rec_lattice.get_cartesian_coords(kpt))
    print('velocity old cartesian transformation')
    print(velocity)
    print()
    print('cartesian with new transformation')
    print(np.dot(MATRIX, kpt.T).T)

    print('velocity')
    # print(abs(_rec_lattice.get_cartesian_coords(de)) / hbar * A_to_m * m_to_cm * Ry_to_eV)
    print(abs(np.dot(MATRIX, de.T).T) / hbar * A_to_m * m_to_cm * Ry_to_eV)

    print('cartesian velocity new transformation')
    velocity = abs(np.dot(dir_matrix, de)) / (hbar * 2 * pi) / 0.52917721067 * A_to_m * m_to_cm * Ry_to_eV

    print('compound mass')
    # comp_mass= hbar ** 2 / (np.dot(MATRIX, np.dot(MATRIX.T, dde))* 4 * pi ** 2) / m_e / A_to_m ** 2 * e * Ry_to_eV # isotropic InN mass but aniso for GaAs
    # comp_mass= hbar ** 2 / (np.dot(MATRIX, np.dot(dde, MATRIX))* 4 * pi ** 2) / m_e / A_to_m ** 2 * e * Ry_to_eV
    # comp_mass= (hbar) ** 2 / np.dot(dir_matrix, np.dot(MATRIX, dde)) /(2*pi)/ m_e / A_to_m ** 2 * e * Ry_to_eV
    comp_mass = 1/(dde/ 0.52917721067) * e / Ry_to_eV / A_to_m**2 * (hbar*2*np.pi)**2 / m_e
    print(comp_mass)
    # print(MATRIX**2)
    # print(np.dot(MATRIX, MATRIX.T))
    # print(np.dot(np.dot(MATRIX, comp_mass), MATRIX))

    mass=0.065
    _, v_poly, effective_mass = get_poly_energy(_rec_lattice.get_cartesian_coords(kpt)
        ,poly_bands=[[
            [[0.0, 0.0, 0.0], [0.0, mass]],
        ]],
        type='n', ib=0,
        bandgap=1.54)

    print('poly velocity with mass={}'.format(mass))
    print(v_poly)
    print('poly mass')
    print(effective_mass)

    quit()

    kmesh = [31,31,31]
    # emesh,dos, nbands, dos_nbands = analytical_bands.get_dos_from_scratch(st,kmesh,-13,20,1000, width=0.05, vbmidx=cbm_bidx-1)
    emesh,dos, nbands = analytical_bands.get_dos_from_scratch(st,kmesh,-13,20,1000, width=0.05)
    plot(emesh, dos)

    # poly_bands = [[[[np.array([ 0.        ,  8.28692586,  0.        ]), np.array([ 0.        , -8.28692586,  0.        ]), np.array([ 3.90649442,  2.76230862,  6.7662466 ]), np.array([-3.90649442, -2.76230862, -6.7662466 ]), np.array([-3.90649442, -2.76230862,  6.7662466 ]), np.array([ 3.90649442,  2.76230862, -6.7662466 ]), np.array([-7.81298883,  2.76230862,  0.        ]), np.array([ 7.81298883, -2.76230862,  0.        ])], [0.0, 0.1]]]]

    # Gamma and X
    # poly_bands = [[[[np.array([ 0.,  0.,  0.])], [1.0, 2.2]]], [[[np.array([ 0.        ,  8.28692586,  0.        ]), np.array([ 0.        , -8.28692586,  0.        ]), np.array([ 3.90649442,  2.76230862,  6.7662466 ]), np.array([-3.90649442, -2.76230862, -6.7662466 ]), np.array([-3.90649442, -2.76230862,  6.7662466 ]), np.array([ 3.90649442,  2.76230862, -6.7662466 ]), np.array([-7.81298883,  2.76230862,  0.        ]), np.array([ 7.81298883, -2.76230862,  0.        ])], [0.0, 0.1]]]]

    # Gamma centered
    poly_bands = [[[[np.array([ 0.,  0.,  0.])], [0.0, 0.2]]]]

    # adding an extra valley at offest of 1 eV
    # poly_bands = [[[[np.array([ 0.        ,  8.28692586,  0.        ]), np.array([ 0.        , -8.28692586,  0.        ]), np.array([ 3.90649442,  2.76230862,  6.7662466 ]), np.array([-3.90649442, -2.76230862, -6.7662466 ]), np.array([-3.90649442, -2.76230862,  6.7662466 ]), np.array([ 3.90649442,  2.76230862, -6.7662466 ]), np.array([-7.81298883,  2.76230862,  0.        ]), np.array([ 7.81298883, -2.76230862,  0.        ])], [0.0, 0.25]]] , [[[np.array([ 0.        ,  8.28692586,  0.        ]), np.array([ 0.        , -8.28692586,  0.        ]), np.array([ 3.90649442,  2.76230862,  6.7662466 ]), np.array([-3.90649442, -2.76230862, -6.7662466 ]), np.array([-3.90649442, -2.76230862,  6.7662466 ]), np.array([ 3.90649442,  2.76230862, -6.7662466 ]), np.array([-7.81298883,  2.76230862,  0.        ]), np.array([ 7.81298883, -2.76230862,  0.        ])], [2, 0.25]]]]

    emesh, dos = get_dos_from_poly_bands(st,_rec_lattice,[6,6,6],-30,30,100000,poly_bands=poly_bands, bandgap=1.54,
                                         width=0.1, SPB_DOS=False, all_values=False)
    plot(emesh,dos)
    # show()

    # this part is not working well:
    #
    # #from a previous calculation of energies
    # engre, latt_points, nwave, nsym, nsymop, symop, br_dir = analytical_bands.get_engre(iband='A')
    # #generate the star functions only one time
    # nstv, vec = analytical_bands.get_star_functions(latt_points,nsym,symop,nwave)
    #
    # energies = []
    # ir_kpts = SpacegroupAnalyzer(st).get_ir_reciprocal_mesh(kmesh)
    # ir_kpts = [k[0] for k in ir_kpts]
    # weights = [k[1] for k in ir_kpts]
    #
    # for kpt in ir_kpts:
    #     energies.append([])
    #     for b in range(len(engre)):
    #         energy = analytical_bands.get_energy(kpt,engre[b], nwave, nsym, nstv, vec)*Ry_to_eV
    #         energies[-1].append(energy)
    #
    # print len(energies),len(energies[0]) #120,21
    #
    # emesh,dos2 = get_dos(energies,weights,-13,20,21)
    # plot(emesh,dos2)
    # show()

