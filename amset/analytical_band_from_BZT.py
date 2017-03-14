# coding: utf-8
from pymatgen.core.units import Energy
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.constants.codata import value as _cd
from math import pi
import numpy as np
from pylab import plot,show

# global constants
hbar = _cd('Planck constant in eV s')/(2*pi)
m_e = _cd('electron mass') # in kg
Ry_to_eV = 13.605698066
A_to_m = 1e-10
m_to_cm = 100
e = _cd('elementary charge')

# TODO: the reading from a fitted band structure file and reproduction of E, dE and d2E should be optimized in speed
# TODO: adding doc to explain each functions their inputs and outputs once Analytical_bands class is optimized.


def outer(v1, v2):
    """returns the outer product of vectors v1 and v2. This is to be used instead of numpy.outer which is ~3x slower."""
    return np.array([[v1[0] * v2[0], v1[0] * v2[1], v1[0] * v2[2]],
                     [v1[1] * v2[0], v1[1] * v2[1], v1[1] * v2[2]],
                     [v1[2] * v2[0], v1[2] * v2[1], v1[2] * v2[2]]])



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



    def get_energy(self, xkpt,engre, nwave, nsym, nstv, vec, vec2=None, out_vec2=None, br_dir=None,cbm=True):
        ' Compute energy for a k-point from star functions '

        sign = -1 if cbm == False else 1
        
        arg = 2*np.pi*vec.dot(xkpt)
        tempc=np.cos(arg)
        spwre=np.sum(tempc,axis=1)-(nsym-nstv)#[:,np.newaxis]
        spwre/=nstv#[:,np.newaxis]
        
        if br_dir is not None:
            dene = np.zeros(3)
            ddene = np.zeros((3,3))
            dspwre = np.zeros((nwave,3))
            ddspwre = np.zeros((nwave,3,3))
            temps=np.sin(arg)
            dspwre=np.sum(vec2*temps[:,:,np.newaxis],axis=1)
            dspwre /= nstv[:,np.newaxis]
    
            #maybe possible a further speed up here
            for nw in xrange(nwave):
                for i in xrange(nstv[nw]):
                    ddspwre[nw] += outer(vec2[nw,i],vec2[nw,i])*(-tempc[nw,i])
                ddspwre[nw] /= nstv[nw]
            #out_tempc = out_vec2*(-tempc[:,:,np.newaxis,np.newaxis])
            #ddspwre = np.sum(out_tempc,axis=1)/ nstv[:,np.newaxis,np.newaxis]
        
        ene=spwre.dot(engre)
        if br_dir is not None:
            dene = np.sum(dspwre.T*engre,axis=1)
            ddene = np.sum(ddspwre*engre.reshape(nwave,1,1)*2,axis=0)
            return sign*ene, dene, ddene
        else:
            return sign*ene



    def get_engre(self,iband=None):
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

        return engre, latt_points, nwave, nsym, nsymop, symop, br_dir



    def get_extreme(self, kpt,iband,only_energy=False,cbm=True):
        engre,latt_points,nwave, nsym, nsymop, symop, br_dir = self.get_engre(iband)
        if only_energy == False:
            energy, denergy, ddenergy = self.get_energy(kpt,engre,nwave, nsym, nstv, vec, vec2, br_dir,cbm=cbm)
            return Energy(energy,"Ry").to("eV"), denergy, ddenergy
        else:
            energy = self.get_energy(kpt,engre,nwave, nsym, nstv, vec, cbm=cbm)
            return Energy(energy,"Ry").to("eV") # This is in eV automatically

        
    def get_dos_from_scratch(self,st,mesh,e_min,e_max,e_points,width=0.2):
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
        engre, latt_points, nwave, nsym, nsymop, symop, br_dir = self.get_engre(iband="A")
        nstv, vec = self.get_star_functions(latt_points,nsym,symop,nwave)
        ir_kpts = SpacegroupAnalyzer(st).get_ir_reciprocal_mesh(mesh)
        ir_kpts = [k[0] for k in ir_kpts]
        weights = [k[1] for k in ir_kpts]
        w_sum = float(sum(weights))
        #weights = [w/w_sum for w in weights]
        #print len(ir_kpts)
        dos = np.zeros(e_range)
        for kpt,w in zip(ir_kpts,weights):
            for b in range(len(engre)):
                e = self.get_energy(kpt,engre[b], nwave, nsym, nstv, vec)*Ry_to_eV
                g = height * np.exp(-((e_mesh - e) / width) ** 2 / 2.)
                dos += w * g
        return e_mesh,dos


    def get_dos(self,energies,weights,e_min,e_max,e_points,width=0.2):
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
    cbm_bidx = 11
    kpts = np.array([[0.5, 0.5, 0.5]])
    coeff_file = 'fort.123'

    analytical_bands = Analytical_bands(coeff_file=coeff_file)
    # read the coefficients file
    engre, latt_points, nwave, nsym, nsymop, symop, br_dir = analytical_bands.get_engre(iband=[cbm_bidx])
    #generate the star functions only one time
    nstv, vec, vec2 = analytical_bands.get_star_functions(latt_points,nsym,symop,nwave,br_dir=br_dir)
    out_vec2 = np.zeros((nwave,nwave,3,3))
    for nw in xrange(nwave):
        for i in xrange(nstv[nw]):
            out_vec2[nw,i]= outer(vec2[nw,i],vec2[nw,i])
            
    # setup
    en, den, dden = [], [], []
    for kpt in kpts:
        energy, de, dde = analytical_bands.get_energy(kpt,engre[0], nwave, nsym, nstv, vec, vec2, out_vec2, br_dir)
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
    st = run.structures[0]

    #dos caclulated on a 15x15x15 mesh of kpoints, 
    #in an energy range [-13,25] eV with 1000 points
    #from get_dos_from_scratch
    kmesh = [15,15,15]
    emesh,dos = analytical_bands.get_dos_from_scratch(st,kmesh,-13,20,1000)
    plot(emesh,dos)
    
    #from a previous calculation of energies
    engre, latt_points, nwave, nsym, nsymop, symop, br_dir = analytical_bands.get_engre(iband='A')
    #generate the star functions only one time
    nstv, vec = analytical_bands.get_star_functions(latt_points,nsym,symop,nwave)

    energies = []
    ir_kpts = SpacegroupAnalyzer(st).get_ir_reciprocal_mesh(kmesh)
    ir_kpts = [k[0] for k in ir_kpts]
    weights = [k[1] for k in ir_kpts]

    for kpt in ir_kpts:
        energies.append([])
        for b in range(len(engre)):
            e = analytical_bands.get_energy(kpt,engre[b], nwave, nsym, nstv, vec)*Ry_to_eV
            energies[-1].append(e)
    
    print len(energies),len(energies[0]) #120,21
    
    emesh,dos2 = analytical_bands.get_dos(energies,weights,-13,20,1000)
    plot(emesh,dos2)
    show()

