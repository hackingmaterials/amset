from numpy import linspace, vstack, polyfit
from pymatgen.electronic_structure.boltztrap import BoltztrapRunner
from pymatgen.io.vasp.outputs import Vasprun, Spin, Procar
from collections import defaultdict
import pymatgen
from math import pi, exp
import warnings
import time

"""This script contains some of the functions used by AMSET such as reading vasp outputs, fitting functions, etc"""

__author__ = "Alireza Faghaninia, Anubhav Jain"
__email__ = "alireza@lbl.gov"

print('pymatgen version:' + pymatgen.__version__)

m_e = 9.10938291e-31        # Electron mass [kg]
e = 1.60217657e-19          # Electron charge [C]
h_planck = 4.135667516e-15  # Planck constant [eV.s]
hbar = h_planck/(2*pi)      # Planck constant divided by 2pi [eV.s]
k_B = 8.6173324e-5          # Boltzmann constant [eV/K]
epsilon_0 = 8.854187817e-12 # Absolute value of dielectric constant in vacuum [C^2/m^2N]

# def write_to_file(file, data, legend=['NA' for i in range(len(data))]):
def write_to_file(file, data, legend):
    """This function takes the opne "file" and each member of the list of lists, "data" to the file with their title
    written above each column read from list of strings, "legend" entered by user"""
    for l in legend:
        file.write("%15s" % l)
    file.write('\n')
    for i in range(len(data[0])):
        for j in range(len(data)):
            file.write("%15.9f" % data[j][i])
        file.write('\n')

def eval_poly_multi(fitted_f, X):
    """This function would unpack a fitted band structure for example and run eval_poly on different sections of it
    that were fitted
    args:
        fitted_f: a fitted fcuntion; format: list of tuples like [(x0, [c0, c1, ..., c7], r20), (...)]
        X: list of floats
    return:
        Y: list with the same type and length as input X
    """
    xidx_old = 0
    Y = []
    for seg in fitted_f:
        xidx = get_idx(seg[0], X)
        Y += eval_poly(seg[1],X[xidx_old:xidx])
        xidx_old = xidx
    # Y += eval_poly(fitted_f[-1][1], X[xidx:])
    Y += [eval_poly(fitted_f[-1][1], X[-1])]
    Y[-1] = Y[-2]   # added this because the previous line generates a large negative number! results doesn't change
    return Y

def eval_poly(coeffs, X):
    """"This function evaluates the value of polynomials with coefficients of "coeff" at each member of X; if X is a
    scalar then the output of the function would also be a scalar
    args:
        coeffs: list of coefficients: for example [2, 0, 1] for 2*x^2+1
        X: list of floats
    """
    n = len(coeffs)-1 # Degree of the polynomial
    if type(X) == list:
        return [sum([coeffs[j] * x ** (n - j) for j in range(n + 1)]) for x in X]
    else:
        return sum([coeffs[j] * X ** (n - j) for j in range(n + 1)])

def get_idx(value, list):
    """get the index of the first occurance of the closes item in the float "list" to the "value" """
    return min(range(len(list)), key=lambda i: abs(list[i] - value))

def avg_xy_data(x, y, normalize=True, tolerance=0.00001):
    """x and y are lists of floats with the same length. This function would first sort these two in a way that the
    first (x) would be increasing and then return a list of data points as tuoles with possibly shorter length after x
    duplicates are averaged out. The output of this function would be ready to plot or to be fed into a curve fitting
    function. The y series are always normalized w.r.t. the first y and the abs value is used to always have a positive
    curve (this is to ensure that the valence band is also increasing). For example:
    avg_xy_data([0, 1, 0] , [0.1, 2, 0.2], normalize=True)
    will return:
    [(0, 0.15), (1, 1.85)]
    """
    idx = sorted(range(len(x)), key=lambda k: x[k])
    x, y = [x[i] for i in idx], [y[i] for i in idx]
    if normalize:
        y = [abs(i - y[0]) for i in y]
    xcopy = []
    ycopy = []
    i = 0
    while i < len(x):
        if i < len(x)-1 and abs(x[i] - x[i+1]) < tolerance:
            sum = y[i]
            counter = 1
            while i < len(x)-1 and abs(x[i] - x[i+1]) < tolerance:
                sum += y[i+1]
                counter += 1
                i += 1
            xcopy.append(x[i])
            ycopy.append(sum/counter)
            i += 1
        else:
            xcopy.append(x[i])
            ycopy.append(y[i])
            i += 1
    return xcopy, ycopy
    # return x, y

def rsquare(data, fitted):
    """Calculates the R2 value of a fit for float lists of the same length: data and fitted"""
    avg = sum(data) / len(data)
    r2 = 1 - sum([(fitted[i] - data[i]) ** 2 for i in range(len(data))]) / sum(
        [(data[i] - avg) ** 2 for i in range(len(data))])
    return r2

def mirror(X, Y):
    X = [-i for i in X[:0:-1]] + X
    Y = Y[:0:-1] + Y
    return X, Y

def curve_fit(xy_data, type='curve'):
    """
    This function fit several nth order polynomial to xy_data and return the coefficients and the x-segments until which
    those polynomials are fitted
    Args:
        xy_data: list of tuples (data points): [(x0, y0), (x1, y1), ... (xn, yn)]
        type: a string that determines the type of fitting, e.g. "band structure"
    return:
        A list of tuples containing the x values to which the polynomial is fitted and coefficients of polynomials and
        the R2 of the fit to that section:
        [(x0, [c0, c1, c2, c3, c4, c5, c6], R2), (x1, [c0.....]) ...]
    """
    time0 = time.time()
    n = 6   # The degree of polynomial. This can also be an optimizable parameter but various tests showed 6 is good.
    X = []
    Y = []
    for i, j in xy_data:
        X.append(i)
        Y.append(j)
    r2_penalty = 10
    nmerg0 = 10
    if len(X)<nmerg0 + 1:
        n = max[len(X)/8, 2]
    division_default = [0.1*X[-1], 0.2*X[-1], 0.5*X[-1], 1.2*X[-1]]
    if len(X) < 3*n:
        raise ValueError('More data points are required for {} fitting'.format(type))
    elif len(X) < len(division_default)*(n+4): # Allowing 4 extra points for a better fit
        division_default = [0.1*X[-1], 0.5*X[-1], 1.2*X[-1]]
    merge_curves = max([0, min([max([nmerg0, len(X)/4-1]), len(X)/len(division_default)-1])])
    if type in ['dos', 'Dos', 'DOS']:
        division_default[0] *= 1.5
    coeffs = [[0 for i in range(n+1)] for j in range(len(division_default))]
    r2 = [0.01 for i in range(len(division_default))]
    x_segs = [0 for i in range(len(division_default))]
    loss_min = 1e30
    step = 0.02 * X[-1]
    # maxiter = 4
    nstep = 5
    division = division_default[:]
    division_selected = division[:]
    # for iter in range(maxiter):
        # step = step/(2**iter)
    for iter in [1]:
        for div0 in range(-nstep,nstep):
            for div1 in range(-nstep, nstep):
                for div2 in range(-nstep, nstep):
                    division[0] = division_default[0] + div0 * step
                    division[1] = division_default[1] + div1 * step
                    division[2] = division_default[2] + div2 * step

# Fitting the first segment that involves mirroring of the data w.r.t. y-axis
                    end_index = get_idx(division[0], X)
                    start_index = 0
                    if end_index - start_index < n + 4:
                        continue
                    X_div0, Y_div0 = mirror(X[start_index:end_index], Y[start_index:end_index])
                    coeffs[0] = polyfit(X_div0, Y_div0, n)
                    r2[0] = rsquare(Y_div0, eval_poly(coeffs[0], X_div0))
                    x_segs[0] = X[end_index]
                    start_index = end_index

# Fitting the rest of the segments
                    for i in range(1,len(division_default)):
                        end_index = get_idx(division[i], X)
                        if end_index - start_index < n + merge_curves + 1:
                            continue
                        if i == len(division_default)-1:
                            merge_curves = 0
                        coeffs[i] = polyfit(X[start_index:end_index], Y[start_index:end_index], n)
                        r2[i] = rsquare(Y[start_index:end_index], eval_poly(coeffs[i], X[start_index:end_index]))
                        x_segs[i] = X[end_index]
                        start_index = end_index

# Checking the discontinuity between the fits
                    loss = 0
                    for i in range(len(division_default)-1):
                        y_left = max([eval_poly(coeffs[i], x_segs[i]), 0.0001]) #   This is to avoid division by zero
                        y_right = eval_poly(coeffs[i+1], x_segs[i])
                        loss = loss + 1/r2[i]**r2_penalty*abs((y_left-y_right)/(y_left+y_right))*exp(-y_left/Y[-1])
                    if loss < loss_min:
                        loss_min = loss
                        division_selected = division[:]
                        coeffs_selected = coeffs[:]
                        r2_selected = r2[:]
                        x_segs_selected = x_segs[:]

    print("The division that resulted in the smoothest {} is:".format(type))
    if len(division_selected) == 4:
        print("%8.5f %8.5f %8.5f %8.5f" % (division_selected[0], division_selected[1], division_selected[2], division_selected[3]))
    elif len(division_selected) == 3:
        print("%8.5f %8.5f %8.5f" % (division_selected[0], division_selected[1], division_selected[2]))
    print("The value of the loss function for fitting a {} is {:>10}".format(type, loss_min))
    print("The R2 values for fitting segments of a {} are:".format(type))
    for r in r2_selected:
        print r
    print("curve_fit time for fitting the {}: {:>7} s".format(type, time.time()-time0))
    return [(x_segs_selected[i], coeffs_selected[i], r2_selected[i]) for i in range(len(r2_selected))]

def fit_dos(free_e):
    """if free_e==False, this function would fit polynomial to the DOS and return that for DOS around the conduction
    band (Ds_n) and the valence band (Ds_p)"""
    if not free_e:
        print('Current version of AMSET does NOT allow for DFT DOS. AMSET will continue as if free_e==True Sorry.')
        Ds_n = Ds_p = None
    else:
        Ds_n = Ds_p = None
    return Ds_n, Ds_p

def fit_procar(isntype, readprocar, n):
    """This function will fit functions to PROCAR and return coefficients of a and c at each point in the k_grid"""
    print('Current version of AMSET does NOT fit orbitals in PROCAR. AMSET will continue with s-like conduction band and p-like valence band assumptions.')
# This needs to be changed to if not readprocar later
    if readprocar or not readprocar:
        if isntype:
            a = [1 for i in range(n)]
            c = [0 for i in range(n)]
        else:
            a = [0 for i in range(n)]
            c = [1 for i in range(n)]
    return a, c

def read_vrun(vrun, elec, hole, plotdir):
    def set_formulation(vrun, dic, bindex, kindex, next_band_inc):
        """"
        args:
        next_band_ing: +1 for conduction bands and -1 for valence bands
        """
        offset = 0.03
        if dic["nbands"] not in [0, 1, 2]:
            raise ValueError("The number of bands in formulation should be either 0 (automatic) or 1/2 for single/coupled-\
                             band formulations")
        if abs(vrun.eigenvalues[(Spin.up, kindex)][bindex + next_band_inc][0] - \
                       vrun.eigenvalues[(Spin.up, kindex)][bindex][0]) < offset:
            if dic["nbands"] == 0:
                print(
                """A coupled-band formulation will be used for electron/hole transport; i.e. elec/hole["nbands"] = 2""")
                dic["nbands"] = 2
            elif dic["nbands"] == 1:
                warnings.warn(
                    """WARNING!!! You are using a single-band formulation, a coupled-band formulation is recommended; i,e, elec/hole["nbands"] = 2""")
        elif dic["nbands"] == 2:
            raise ValueError(
                """A coupled-band formulation does NOT apply to this band structure, try again with elec/hole["nbands"] = 0""")
        else:
            dic["nbands"] = 1
            print("""A single-band formulation is used; i.e. elec/hole["nbands"] = 1""")
        return dic
    def get_xy_band(vrun, recip_kpoints, bindex, kref):
        bandev = [vrun.eigenvalues[(Spin.up, i)][bindex][0] for i in range(len(vrun.actual_kpoints))]
        kdistance = [10*(sum([(k[j] - kref[j])**2 for j in range(3)]))**0.5 for k in recip_kpoints] # 10x is to convert 1/A to 1/nm
        kdistance, bandev = avg_xy_data(kdistance, bandev)
        xy_band = [(kdistance[i], bandev[i]) for i in range(len(kdistance))]
        return xy_band

    volume = vrun.final_structure.volume
    density = vrun.final_structure.density
    lattice = vrun.lattice_rec.matrix / (2 * pi)
    bs = vrun.get_band_structure()
    vbm = bs.get_vbm()
    cbm = bs.get_cbm()
    vbm_bindex = vbm["band_index"][Spin.up][-1]
    vbm_kindex = vbm["kpoint_index"][0]
    cbm_bindex = cbm["band_index"][Spin.up][0]
    cbm_kindex = cbm["kpoint_index"][0]
    vbm_k = vrun.actual_kpoints[vbm_kindex]
    cbm_k = vrun.actual_kpoints[cbm_kindex]
    print('index of last valence band = ' + str(vbm_bindex))
    print('index of first conduction band = ' + str(cbm_bindex))
    recip_kpoints = [[2*pi*sum([k[i] * lattice[i][j] for i in range(3)]) for j in range(3)] for k in vrun.actual_kpoints]
    elec = set_formulation(vrun, elec, cbm_bindex, cbm_kindex, next_band_inc=+1)
    hole = set_formulation(vrun, hole, vbm_bindex, vbm_kindex, next_band_inc=-1)
    xy_cond_band = get_xy_band(vrun, recip_kpoints, cbm_bindex, cbm_k)
    fitted_cond_band = curve_fit(xy_cond_band, 'band structure') # output has the format [(x, [], x),(...), ...] see curve_fit
    if elec["nbands"] == 2:
        xy_cond_band2 = get_xy_band(vrun, recip_kpoints, cbm_bindex+1, cbm_k)
        fitted_cond_band2 = curve_fit(xy_cond_band2, 'band structure')

    xy_val_band = get_xy_band(vrun, recip_kpoints, vbm_bindex, vbm_k)
    fitted_val_band = curve_fit(xy_val_band, 'band structure')
    if hole["nbands"] == 2:
        xy_val_band2 = get_xy_band(vrun, recip_kpoints, vbm_bindex-1, vbm_k)
        fitted_val_band2 = curve_fit(xy_val_band2, 'band structure')


    # To calculate the 2nd derivative hence the effective mass we use the coefficient of x**2 times 2
    if elec["m"] == 0:
        elec["m"] = 1 / (fitted_cond_band[0][1][-3] * 2 / (hbar**2) * 1e-18 / e * m_e)
    print fitted_cond_band[0][1]
    if elec["m2"] == 0 and elec["nbands"] == 2:
        elec["m2"] = 1 / (fitted_cond_band2[0][1][-3] * 2 / hbar ** 2 * 1e-18 / e * m_e)
    if hole["m"] == 0:
        hole["m"] = 1 / (fitted_val_band[0][1][-3] * 2 / hbar ** 2 * 1e-18 / e * m_e)
    if hole["m2"] == 0 and hole["nbands"] == 2:
        hole["m2"] = 1 / (fitted_val_band2[0][1][-3] * 2 / hbar ** 2 * 1e-18 / e * m_e)


    with open(plotdir + 'DFT_bands.txt', 'w') as f:
        write_to_file(f, data=[[i[0] for i in xy_cond_band],[i[1] for i in xy_cond_band]], legend=["k-1/nm", "cond-eV"])


# End of read_vrun function
    return volume, density, lattice, elec, hole, fitted_cond_band



if __name__ == "__main__":

# The following is for extracting DOS and PROCAR and their fitting (e.g. free_e=False) and will be added to AMSET later
    print('coordinates of vbm: ' + str(vrun.actual_kpoints[vbm_kindex]))

    dos_e = vrun.complete_dos.energies
    print(len(dos_e))
    print(dos_e[0:25])
    dos = vrun.complete_dos.densities
    print(len(dos[Spin.up]))
    print(dos[Spin.up][0:25])
    procar = Procar('ZnS_tests/PROCAR')

    or_data = procar.data
    # procar.data; spin: nd.array accessed with (k-point index, band index, ion index, orbital index)
    print(st.num_sites)
    print(procar.orbitals)
    print(or_data[Spin.up][vbm_kindex][vbm_bindex][:][:])
    vbm_s = sum([or_data[Spin.up][vbm_kindex][vbm_bindex][i][0] for i in range(st.num_sites)])
    vbm_p = sum([or_data[Spin.up][vbm_kindex][vbm_bindex][i][j] for i in range(st.num_sites) for j in range(1,4)])

    cbm_s = sum([or_data[Spin.up][cbm["kpoint_index"][0]][cbm["band_index"][Spin.up][0]][i][0] for i in range(st.num_sites)])
    cbm_p = sum([or_data[Spin.up][cbm["kpoint_index"][0]][cbm["band_index"][Spin.up][0]][i][j] for i in range(st.num_sites) for j in range(1,4)])
    print('cbm s orbital: ' + str(cbm_s))
    print('cbm p orbital: ' + str(cbm_p))


    eigen = defaultdict(dict)
    for (spin, index), values in vrun.eigenvalues.items():
        eigen[index][str(spin)] = values
    ## Now egein's first index is band index and 2nd is spin which is either "1" or "-1", 2nd to last idx is band idx and the last is either 0 or 1 and 0 is energy and 1 is occupation
    ## e.g. print(eigen[3]["-1"][3][0])
    print('value of the VBM: ' + str(eigen[vbm_kindex]["1"][vbm_bindex][0]))
    print('value of the CBM: ' + str(eigen[cbm["kpoint_index"][0]]["1"][cbm["band_index"][Spin.up][0]][0]))

