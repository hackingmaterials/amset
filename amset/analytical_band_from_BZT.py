# coding: utf-8
from pymatgen.core.units import Energy
from scipy.constants.codata import value as _cd
from math import pi
import numpy as np

# global constants
hbar = _cd('Planck constant in eV s')/(2*pi)
m_e = _cd('electron mass') # in kg
Ry_to_eV = 13.605698066
A_to_m = 1e-10
m_to_cm = 100
e = _cd('elementary charge')

# TODO: the reading from a fitted band structure file and reproduction of E, dE and d2E should be optimized in speed
# TODO: adding doc to explain each functions their inputs and outputs once Analytical_bands class is optimized.

class Analytical_bands(object):
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
        if br_dir != None:
            vec2 = np.zeros((nwave,nsym,3))

        for nw in xrange(nwave):
            nstv[nw], vec[nw]  = self.stern(latt_points[nw],nsym,symop)
            if br_dir != None:
                vec2[nw] = vec[nw].dot(br_dir)
        #print vec
        if br_dir!= None:
            return nstv, vec, vec2
        else:
            return nstv, vec

    def get_energy(self, xkpt,engre, nwave, nsym, nstv, vec, vec2=None, br_dir=None,cbm=True):
        ' Compute energy for a k-point from star functions '

        sign = -1 if cbm == False else 1
        
        arg = 2*np.pi*vec.dot(xkpt)
        tempc=np.cos(arg)
        spwre=np.sum(tempc,axis=1)-(nsym-nstv)
        spwre/=nstv
        
        if br_dir != None:
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
                    ddspwre[nw] += np.outer(vec2[nw,i],vec2[nw,i])*(-tempc[nw,i])
                ddspwre[nw] /= nstv[nw]
        
        ene=spwre.dot(engre)
        if br_dir != None:
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
            engre=np.zeros(nwave)
            for i,l in enumerate(f):
                if i<nwave:
                    latt_points[i]=np.fromstring(l,sep=' ')
                elif i == nwave:
                    bmin,bmax=np.fromstring(l,sep=' ',dtype=int)
                    if iband == None:
                        print "Bands range: {}-{}".format(bmin,bmax)
                        break
                    elif iband > bmax or iband < bmin:
                        print "ERROR! iband not in range : {}-{}".format(bmin,bmax)
                        return
                    iband2 = iband-bmin+1
                elif i == nwave+iband2:
                    engre=np.fromstring(l,sep=' ')
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

if __name__ == "__main__":
    # user inputs
    cbm_bidx = 11
    kpts = np.array([[0.5, 0.5, 0.5]])
    coeff_file = 'fort.123'

    analytical_bands = Analytical_bands(coeff_file=coeff_file)
    # read the coefficients file
    engre, latt_points, nwave, nsym, nsymop, symop, br_dir = analytical_bands.get_engre(iband=cbm_bidx)
    #generate the star functions only one time
    nstv, vec, vec2 = analytical_bands.get_star_functions(latt_points,nsym,symop,nwave,br_dir=br_dir)
    # setup
    en, den, dden = [], [], []
    for kpt in kpts:
        energy, de, dde = analytical_bands.get_energy(kpt,engre, nwave, nsym, nstv, vec, vec2, br_dir)
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





