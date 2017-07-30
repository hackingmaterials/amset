# coding: utf-8
from pymatgen.core.units import Energy
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.constants.codata import value as _cd
from math import pi
import numpy as np
import warnings
from pylab import plot,show, scatter


# global constants
hbar = _cd('Planck constant in eV s')/(2*pi)
m_e = _cd('electron mass') # in kg
Ry_to_eV = 13.605698066
A_to_m = 1e-10
A_to_nm = 0.1
m_to_cm = 100
e = _cd('elementary charge')

# TODO: the reading from a fitted band structure file and reproduction of E, dE and d2E should be optimized in speed
# TODO: adding doc to explain each functions their inputs and outputs once Analytical_bands class is optimized.


__author__ = "Francesco Ricci and Alireza Faghaninia"
__copyright__ = "Copyright 2017, HackingMaterials"
__maintainer__ = "Francesco Ricci"



def norm(v):
    """method to quickly calculate the norm of a vector (v: 1x3 or 3x1) as numpy.linalg.norm is slower for this case"""
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5



def outer(v1, v2):
    """returns the outer product of vectors v1 and v2. This is to be used instead of numpy.outer which is ~3x slower."""
    return np.array([[v1[0] * v2[0], v1[0] * v2[1], v1[0] * v2[2]],
                     [v1[1] * v2[0], v1[1] * v2[1], v1[1] * v2[2]],
                     [v1[2] * v2[0], v1[2] * v2[1], v1[2] * v2[2]]])



def get_poly_energy(kpt, poly_bands, type, ib=0, bandgap=1, all_values = False):
    """

    :param kpt (list): coordinates of a given k-point in the actual cartesian coordinates and NOT fractional coordinates
    :param rotations: symmetry rotation operations
    :param translations: symmetry translational operations
    :param poly_bands [[lists]]: each member of the first list represents a band: in each band a list of lists of lists
        should contain a list of two-member lists: the two members are: the coordinates of extrema k-point and its
        symmetrically equivalent points and another two members list of
        [first member: energy offset from the main extremum (i.e. CBM/VBM), second member: the effective mass]
        example poly_bands = [[ [[[0.5, 0.5, 0.5]], [0, 0.1]], [[[0, 0, 0]], [0.5, 0.2]]]] represents a
        band structure with a single band; this parabolic band (hbar**2k**2/2m*) at point X
        where k is norm(k-[0.5,0.5,0.5]) if the k-point is closer to X than Gamma. Additionally, this band has
        another extremum at an energy level 0.5 eV above the first/main extremum/CBM. If type is "p" the band structure
        would be a mirror image. VBM is always set to 0.0 eV and the CBM is set to the bandgap

    :param type (str): "n" or "p"
    :param ib (int): the band index, 0 is for the first band and maximum allowed value is len(poly_bands)-1
    :param bandgap:
    :return:
    """
    # determine the sign of energy from type; e.g. p-type energies are negative with VBM=0.0
    kpt = np.array(kpt)
    sgn = (-1)**(["p", "n"].index(type)+1)
    band_shapes = poly_bands[ib]
    min_kdistance = 1e32
    energy_list = []
    for ks, c in band_shapes: #ks are all symmetrically equivalent k-points to the extremum k that are in the first BZ
        for k in ks:
        # distance = min([norm(kpt-k) for k in ks])
        #     distance = norm(kpt-k)/(2*pi)
            distance = norm(kpt-k)
            energy_list.append(bandgap * ["p", "n"].index(type) +\
                               sgn * (c[0] + hbar ** 2 * (distance) ** 2 / (2 * m_e * c[1]) * e * 1e18))
            # for k in ks:
            #     distance = norm(kpt-k)
            if distance < min_kdistance:
                min_kdistance = distance
                coefficients = c


    # coefficients[0] is the constant
    # coefficient[1] is the effective mass
    eff_m = coefficients[1]
    # energy = bandgap*["p", "n"].index(type) # if p-type we do not add any value to the calculated energy for the band gap
    energy = sgn * bandgap/2.0
    energy += sgn*(coefficients[0] + hbar**2 * min_kdistance**2 / (2*m_e*eff_m) * e*1e18) # last part is unit conv. to eV
    # print hbar**2 * min_kdistance**2 / (2*m_e*eff_m) * e*1e18
    v = hbar*min_kdistance/(m_e*eff_m) *1e11*e # last part is unit conversion to cm/s

    if not all_values:
        return energy, np.array([v, v, v]), sgn*np.array([[eff_m, 0.0, 0.0], [0.0, eff_m, 0.0], [0.0, 0.0, eff_m]])
    else:
        return energy_list, min_kdistance
    # de = 0
    # dde = 0
    # for i, c in enumerate(coefficients):
    #     energy += c * min_kdistance**i * sgn
    #     if i > 0:
    #         de += i * c *min_kdistance**(i-1)
    #         if i > 1:
    #             dde += i * (i-1) * c * min_kdistance ** (i - 2)
    # de = np.array([de, de, de])
    # dde = np.array([[dde, 0.0, 0.0],
    #                 [0.0, dde, 0.0],
    #                 [0.0, 0.0, dde] ])
    # return energy, de, dde

def get_energy(xkpt, engre, nwave, nsym, nstv, vec, vec2=None, out_vec2=None, br_dir=None, cbm=True):
    ' Compute energy for a k-point from star functions '

    sign = -1 if cbm == False else 1
    arg = 2 * np.pi * vec.dot(xkpt)
    tempc = np.cos(arg)
    spwre = np.sum(tempc, axis=1) - (nsym - nstv)  # [:,np.newaxis]
    spwre /= nstv  # [:,np.newaxis]

    if br_dir is not None:
        dene = np.zeros(3)
        ddene = np.zeros((3, 3))
        dspwre = np.zeros((nwave, 3))
        ddspwre = np.zeros((nwave, 3, 3))
        temps = np.sin(arg)
        dspwre = np.sum(vec2 * temps[:, :, np.newaxis], axis=1)
        dspwre /= nstv[:, np.newaxis]

        # maybe possible a further speed up here
        # for nw in xrange(nwave):
        # for i in xrange(nstv[nw]):
        # ddspwre[nw] += outer(vec2[nw,i],vec2[nw,i])*(-tempc[nw,i])
        # ddspwre[nw] /= nstv[nw]
        out_tempc = out_vec2 * (-tempc[:, :, np.newaxis, np.newaxis])
        ddspwre = np.sum(out_tempc, axis=1) / nstv[:, np.newaxis, np.newaxis]

    ene = spwre.dot(engre)
    if br_dir is not None:
        dene = np.sum(dspwre.T * engre, axis=1)
        ddene = np.sum(ddspwre * engre.reshape(nwave, 1, 1) * 2, axis=0)
        return sign * ene, dene, ddene
    else:
        return sign * ene



def get_dos_from_poly_bands(st, lattice_matrix, mesh, e_min, e_max, e_points, poly_bands, bandgap, width=0.1, SPB_DOS=False, all_values=False):
        '''
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
        '''

        height = 1.0 / (width * np.sqrt(2 * np.pi))
        e_mesh, step = np.linspace(e_min, e_max,num=e_points, endpoint=True, retstep=True)

        e_range = len(e_mesh)
        ir_kpts_n_weights = SpacegroupAnalyzer(st).get_ir_reciprocal_mesh(mesh)
        ir_kpts = [k[0] for k in ir_kpts_n_weights]
        weights = [k[1] for k in ir_kpts_n_weights]

        # ir_kpts_base = [k for k in ir_kpts]
        # counter = 1
        # for x in range(1,1):
        #     for y in range(1,1):
        #         for z in range(1,1):
        #             counter += 1
        #             ir_kpts += [k+[x, y, z] for k in ir_kpts_base]
        # weights *= counter

        ir_kpts = [lattice_matrix.get_cartesian_coords(k)/A_to_nm for k in ir_kpts]

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
    '''
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
    '''
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
        ' Compute star function for a specific g vector '
        
        trial = symop[:nsym].dot(g)
        stg = np.unique(trial.view(np.dtype((np.void, trial.dtype.itemsize*trial.shape[1])))).view(trial.dtype).reshape(-1, trial.shape[1])
        nst = len(stg)
        stg = np.concatenate((stg,np.zeros((nsym-nst,3))))
        return nst, stg



    def get_star_functions(self, latt_points, nsym,symop,nwave,br_dir=None):
        ' Compute star functions for all R vectors and symmetries '

        nstv = np.zeros(nwave,dtype='int')
        vec = np.zeros((nwave,nsym,3))
        if br_dir is not None:
            vec2 = np.zeros((nwave,nsym,3))

        for nw in xrange(nwave):
            nstv[nw], vec[nw]  = self.stern(latt_points[nw],nsym,symop)
            if br_dir is not None:
                vec2[nw] = vec[nw].dot(br_dir)

        #print vec
        if br_dir is not None:
            return nstv, vec, vec2
        else:
            return nstv, vec


    # TODO: I copied this function outside of the class so that I can take advantage of Parallel function, remove it from here later! (it doesn't seem like it's using anything from the class anyway!)
    # def get_energy(self, xkpt,engre, nwave, nsym, nstv, vec, vec2=None, out_vec2=None, br_dir=None,cbm=True):
    #     ' Compute energy for a k-point from star functions '
    #
    #     sign = -1 if cbm == False else 1
    #     arg = 2*np.pi*vec.dot(xkpt)
    #     tempc=np.cos(arg)
    #     spwre=np.sum(tempc,axis=1)-(nsym-nstv)#[:,np.newaxis]
    #     spwre/=nstv#[:,np.newaxis]
    #
    #     if br_dir is not None:
    #         dene = np.zeros(3)
    #         ddene = np.zeros((3,3))
    #         dspwre = np.zeros((nwave,3))
    #         ddspwre = np.zeros((nwave,3,3))
    #         temps=np.sin(arg)
    #         dspwre=np.sum(vec2*temps[:,:,np.newaxis],axis=1)
    #         dspwre /= nstv[:,np.newaxis]
    #
    #         #maybe possible a further speed up here
    #         #for nw in xrange(nwave):
    #             #for i in xrange(nstv[nw]):
    #                 #ddspwre[nw] += outer(vec2[nw,i],vec2[nw,i])*(-tempc[nw,i])
    #             #ddspwre[nw] /= nstv[nw]
    #         out_tempc = out_vec2*(-tempc[:,:,np.newaxis,np.newaxis])
    #         ddspwre = np.sum(out_tempc,axis=1)/ nstv[:,np.newaxis,np.newaxis]
    #
    #     ene=spwre.dot(engre)
    #     if br_dir is not None:
    #         dene = np.sum(dspwre.T*engre,axis=1)
    #         ddene = np.sum(ddspwre*engre.reshape(nwave,1,1)*2,axis=0)
    #         return sign*ene, dene, ddene
    #     else:
    #         return sign*ene



    def get_engre(self,iband=None, return_bmin=False):
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
                    bmin,bmax=np.fromstring(l,sep=' ',dtype=int)
                    if iband == 'A':
                        iband = range(bmin,bmax+1)
                        # print "bmin: {}".format(bmin)
                        # print "bmax: {}".format(bmax)

                    elif iband == None:
                        print "Bands range: {}-{}".format(bmin,bmax)
                        break
                    elif max(iband) > bmax or min(iband) < bmin:
                        print "ERROR! iband not in range : {}-{}".format(bmin,bmax)
                        return
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
        '''
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
        '''


        height = 1.0 / (width * np.sqrt(2 * np.pi))
        e_mesh, step = np.linspace(e_min, e_max,num=e_points, endpoint=True, retstep=True)
        e_range = len(e_mesh)
        cbm_new_idx = None

        if vbmidx:
            engre, latt_points, nwave, nsym, nsymop, symop, br_dir, bmin = self.get_engre(iband="A", return_bmin=True)
            cbm_new_idx = vbmidx - bmin + 1
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
        #print len(ir_kpts)
        dos = np.zeros(e_range)
        # for b in range(len(engre)):
        #     energy = get_energy([0.0, 0.0, 0.0], engre[b], nwave, nsym, nstv, vec) * Ry_to_eV
        #     print energy
        for kpt,w in zip(ir_kpts,weights):
            for b in range(len(engre)):
                energy = get_energy(kpt,engre[b], nwave, nsym, nstv, vec)*Ry_to_eV
                if b >= cbm_new_idx:
                    energy += scissor/2
                else:
                    energy -= scissor/2
                g = height * np.exp(-((e_mesh - energy) / width) ** 2 / 2.)
                dos += w * g
        if vbmidx:
            return e_mesh, dos, len(engre), bmin
        else:
            return e_mesh,dos, len(engre)


    def get_dos_standard(self,energies,weights,e_min,e_max,e_points,width=0.2):
        '''
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
        '''
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
    cbm_bidx = 5
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

    kpts = [[ 0.,  0.,  0.], [ 0.42105263,  0.42105263,  0.        ]] # Si kVBM and kCBM respectively
    # coeff_file = '../test_files/PbTe/fort.123'
    # coeff_file = "../test_files/GaAs/fort.123_GaAs_1099kp"
    coeff_file = "../test_files/Si/Si_fort.123"
    analytical_bands = Analytical_bands(coeff_file=coeff_file)
    # read the coefficients file
    engre, latt_points, nwave, nsym, nsymop, symop, br_dir = analytical_bands.get_engre(iband=[cbm_bidx])
    #generate the star functions only one time
    nstv, vec, vec2 = analytical_bands.get_star_functions(latt_points,nsym,symop,nwave,br_dir=br_dir)
    out_vec2 = np.zeros((nwave,max(nstv),3,3))
    for nw in xrange(nwave):
        for i in xrange(nstv[nw]):
            out_vec2[nw,i]= outer(vec2[nw,i],vec2[nw,i])
            
    # setup
    en, den, dden = [], [], []
    for kpt in kpts:
        energy, de, dde = get_energy(kpt,engre[0], nwave, nsym, nstv, vec, vec2, out_vec2, br_dir)
        en.append(energy*Ry_to_eV)
        den.append(de)
        dden.append(dde*2*pi)

    print("outputs:")
    print("Energy: {}".format(en))
    print("1st derivate:")
    print den
    print("2nd derivative:")
    print dde
    m_tensor = hbar ** 2 /(dde*4*pi**2) / m_e / A_to_m ** 2 * e * Ry_to_eV # m_tensor: the last part is unit conversion
    print("effective mass tensor")
    print(m_tensor)


    print("group velocity:")
    v = de /hbar*A_to_m*m_to_cm * Ry_to_eV # to get v in units of cm/s
    print v

    run = Vasprun('vasprun.xml')
    lattice_matrix = run.final_structure.lattice.reciprocal_lattice
    st = run.structures[0]


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

    emesh, dos = get_dos_from_poly_bands(st,lattice_matrix,[6,6,6],-30,30,100000,poly_bands=poly_bands, bandgap=1.54,
                                         width=0.1, SPB_DOS=False, all_values=False)
    plot(emesh,dos)
    show()

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

