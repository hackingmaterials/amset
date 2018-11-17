import numpy as np
import warnings

from amset.utils.analytical_band_from_bzt1 import Analytical_bands, get_energy
from amset.utils.band_structure import kpts_to_first_BZ, get_bindex_bspin, \
    get_closest_k
from amset.utils.constants import Ry_to_eV, hbar, A_to_m, m_to_cm, e, m_e, \
    Hartree_to_eV
from amset.utils.detect_peaks import detect_peaks
from amset.utils.general import outer, AmsetError, norm
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath

try:
    import BoltzTraP2
    import BoltzTraP2.dft
    from BoltzTraP2 import sphere, fite
except ImportError:
    warnings.warn('BoltzTraP2 not imported; "boltztrap2" interpolation not available.')


def get_energy_args(coeff_file, ibands):
    """
    Args:
        coeff_file (str): the address to the cube (*.123) file
        ibands ([int]): list of band numbers to be calculated; note that the
            first band index is 1 not 0

    Returns (tuple): necessary inputs for calc_analytical_energy or get_energy
    """
    analytical_bands = Analytical_bands(coeff_file=coeff_file)
    try:
        engre, latt_points, nwave, nsym, nsymop, symop, br_dir = \
            analytical_bands.get_engre(iband=ibands)
    except TypeError as e:
        raise ValueError('try reducing max_Ecut to include fewer bands', e)

    nstv, vec, vec2 = analytical_bands.get_star_functions(
            latt_points, nsym, symop, nwave, br_dir=br_dir)
    out_vec2 = np.zeros((nwave, max(nstv), 3, 3))
    for nw in range(nwave):
        for i in range(nstv[nw]):
            out_vec2[nw, i] = outer(vec2[nw, i], vec2[nw, i])
    return engre, nwave, nsym, nstv, vec, vec2, out_vec2, br_dir


def interpolate_bs(kpts, interp_params, iband, sgn=None, method="boltztrap1",
                   scissor=0.0, matrix=None, n_jobs=1, return_mass=True):
    """
    Args:
        kpts ([1x3 array]): list of fractional coordinates of k-points
        interp_params (tuple): a tuple or list containing positional
            arguments fed to the interpolation method.
            e.g. for boltztrap1:
                engre, nwave, nsym, stv, vec, vec2, out_vec2, br_dir
            and for boltztrap2:
                (equivalences, lattvec, coeffs)
        iband (int): the band index for which the list of energy, velocity
            and mass is returned. If "boltztrap2" method is used, this is the
            actual band index while if "boltztrap1" methid is used, this is the
            ith band among the bands that were included in the fit (i.e. when
            get_energy_args is called)
        sgn (float): options are +1 for valence band and -1 for conduction bands
            sgn is basically ignored (doesn't matter) if scissor==0.0
        method (str): the interpolation method. Current options are
            "boltztrap1", "boltztrap2"
        scissor (float): the amount by which the band gap is modified/scissored
        matrix (3x3 np.ndarray): the direct lattice matrix used to convert
            the velocity (in fractional coordinates) to cartesian in
            boltztrap1 method.
        n_jobs (int): number of processes used in boltztrap1 interpolation
        return_mass (bool): whether to return the effective mass values or not

    Returns (tuple of energies, velocities, masses lists/np.ndarray):
        energies ([float]): energy values at kpts for a corresponding iband
        velocities ([3x1 array]): velocity vectors
        masses ([3x3 matrix]): list of effective mass tensors
    """
    #TODO: effective mass is still inconsistent between btp1 and btp2 w/o any transformation used since it is not used in Amset ok but has to be checked with the right transformation
    if matrix is None:
        matrix = np.eye(3)
    if not sgn:
        if scissor == 0.0:
            sgn=0.0
        else:
            raise ValueError('To apply scissor "sgn" is required: -1 or +1')
    masses = []
    if method=="boltztrap1":
        engre, nwave, nsym, nstv, vec, vec2, out_vec2, br_dir = interp_params
        energies = []
        velocities = []
        if n_jobs == 1:
            results = []
            for kpt in kpts:
                result = get_energy(kpt, engre[iband], nwave, nsym,
                                             nstv, vec, vec2, out_vec2, br_dir,
                                             return_dde=return_mass)
                results.append(result)
        else:
            inputs = [(kpt, engre[iband], nwave, nsym, nstv, vec, vec2,
                                            out_vec2, br_dir) for kpt in kpts]
            with Pool(n_jobs if n_jobs != -1 else cpu_count()) as p:
                results = p.starmap(get_energy, inputs)
        for result in results:
            energy = result[0] * Ry_to_eV - sgn * scissor / 2.0
            velocity = abs(np.dot(matrix/np.linalg.norm(matrix), result[1])) / hbar / 0.52917721067 * A_to_m * m_to_cm * Ry_to_eV
            if return_mass:
                effective_m = 1/(result[2]/ 0.52917721067**2*Ry_to_eV) * e / A_to_m**2 * hbar**2 / m_e
                masses.append(effective_m)
            energies.append(energy)
            velocities.append(velocity)
    elif method=="boltztrap2":
        if n_jobs != 1:
            warnings.warn('n_jobs={}: Parallel not implemented w/ boltztrap2'
                          .format(n_jobs))
        equivalences, lattvec, coeffs = interp_params
        fitted = fite.getBands(np.array(kpts), equivalences, lattvec, coeffs,
                               curvature=return_mass)
        energies = fitted[0][iband - 1] * Hartree_to_eV - sgn * scissor / 2.
        velocities = abs(np.matmul(matrix/np.linalg.norm(matrix), fitted[1][:, iband - 1, :]).T) * Hartree_to_eV / hbar * A_to_m * m_to_cm / 0.52917721067
        if return_mass:
            masses = 1/(fitted[2][:, :, iband - 1, :].T/ 0.52917721067**2*Hartree_to_eV)* e / A_to_m**2 * hbar**2/m_e
    else:
        raise AmsetError("Unsupported interpolation method: {}".format(method))
    if return_mass:
        return energies, velocities, masses
    else:
        return energies, velocities


def get_dos_boltztrap2(params, st, mesh, estep=0.001, vbmidx=None,
                       width=0.2, scissor=0.0):
    """
    Calculates the density of states (DOS) based on boltztrap2 interpolation.

    Args:
        params (list/tuple): parameters required for boltztrap2 interpolation
        st (pymatgen Structure object): required for generating irriducible
            brillouin zone mesh)
        mesh (a 3x1 list or np.ndarray): the k-grid; e.g. [13, 15, 11]
        estep (float): small energy step, the smaller better but more expensive
        vbmidx (int): the index of the valence band maximum assuming the index
            of the first band is 0
        width (float): energy bandwidth/smearing parameter.
        scissor (float): the intended change to the current band gap

    Returns (tuple): in the same order: 1) list of enegy values 2) list of
        densities at those energy values and 3) number of bands considered
    """
    (equivalences, lattvec, coeffs) = params
    ir_kpts = SpacegroupAnalyzer(st).get_ir_reciprocal_mesh(mesh)
    ir_kpts = [k[0] for k in ir_kpts]
    weights = [k[1] for k in ir_kpts]
    w_sum = float(sum(weights))
    weights = [w / w_sum for w in weights]

    energies, _ = fite.getBands(np.array(ir_kpts), equivalences=equivalences,
                                lattvec=lattvec, coeffs=coeffs)
    energies *= Hartree_to_eV  # shape==(bands, nkpoints)
    nbands = energies.shape[0]
    if vbmidx:
        ib_start = max(0, vbmidx-4)
        ib_end = min(energies.shape[0], vbmidx+1+4)
        energies[vbmidx + 1:, :] += scissor / 2.
        energies[:vbmidx + 1, :] -= scissor / 2.
        energies = energies[ib_start:ib_end, :]
        nbands = ib_end - ib_start
    e_min = np.min(energies)
    e_max = np.max(energies)
    height = 1.0 / (width * np.sqrt(2 * np.pi))
    e_points = int(round((e_max - e_min) / estep))
    e_mesh, step = np.linspace(e_min, e_max, num=e_points, endpoint=True,
                               retstep=True)
    e_range = len(e_mesh)
    dos = np.zeros(e_range)

    for ik, w in enumerate(weights):
        for b in range(nbands):
            g = height * np.exp(-((e_mesh - energies[b, ik]) / width) ** 2 / 2.)
            dos += w * g
    return e_mesh, dos, nbands


def get_bs_extrema(bs, coeff_file=None, interp_params=None, method="boltztrap1",
                   line_density=30, min_normdiff=4.0,
                   Ecut=None, eref=None, return_global=False, n_jobs=-1,
                   nbelow_vbm=0, nabove_cbm=0, scissor=0.0):
    """
    Returns a dictionary of p-type (valence) and n-type (conduction) band
        extrema k-points by looking at the 1st and 2nd derivatives of the bands

    Args:
        bs (pymatgen BandStructure object): must contain Structure and have
            the same number of valence electrons and settings as the vasprun.xml
            from which coeff_file is generated.
        coeff_file (str): path to the cube file from BoltzTraP run
        line_density (int): maximum number of k-points between each two
            consecutive high-symmetry k-points
        v_cut (float): threshold under which the derivative is assumed 0 [cm/s]
        min_normdiff (float): the minimum allowed distance
            norm(cartesian k in 1/nm) in extrema; this is important to avoid
            numerical instability errors or finding peaks that are too close
            to each other for Amset formulation to be relevant.
        Ecut (float or dict): max energy difference with CBM/VBM allowed for
            extrema. Valid examples: 0.25 or {'n': 0.5, 'p': 0.25} , ...
        eref (dict): BandStructure global VBM/CBM used as a global reference
            energy for Ecut. Example: {'n': 6.0, 'p': 5.0}. Ignored if None in
            which case maximum/minimum of the current valence/conduction used
        return_global (bool): in addition to the extrema, return the actual
            CBM (global minimum) and VBM (global maximum) w/ their k-point
        n_jobs (int): number of processors used in boltztrap1 interpolation
        nbelow_vbm (int): # of bands below the last valence band
        nabove_vbm (int): # of bands above the first conduction band
        scissor (float): the amount by which the band gap is altered/scissored.

    Returns (dict): {'n': list of extrema fractional coordinates, 'p': same}
    """
    lattice_matrix = bs.structure.lattice.reciprocal_lattice.matrix
    def to_cart(k):
        """
        convert fractional k-points to cartesian coordinates in (1/nm) units
        """
        return np.dot(lattice_matrix, k)*10.
    Ecut = Ecut or 10 * k_B * 300
    if not isinstance(Ecut, dict):
        Ecut = {'n': Ecut, 'p': Ecut}
    if eref is None:
        global_extrema = {'n': {}, 'p': {}}
    else:
        cbmk = np.array(bs.get_cbm()['kpoint'].frac_coords)
        vbmk = np.array(bs.get_vbm()['kpoint'].frac_coords)
        global_extrema = {'n': {'energy': eref['n'], 'kpoint': cbmk},
                          'p': {'energy': eref['p'], 'kpoint': vbmk}}
    final_extrema = {'n': [], 'p': []}
    hsk = HighSymmKpath(bs.structure)

    hs_kpoints, _ = hsk.get_kpoints(line_density=line_density)
    hs_kpoints = kpts_to_first_BZ(hs_kpoints)
    vbm_idx, _ = get_bindex_bspin(bs.get_vbm(), is_cbm=False)
    cbm_idx, _ = get_bindex_bspin(bs.get_cbm(), is_cbm=True)

    if method == "boltztrap1" and interp_params is None:
        interp_params = get_energy_args(coeff_file=coeff_file,
                                        ibands=[vbm_idx + 1 - nbelow_vbm,
                                                cbm_idx + 1 + nabove_cbm])

    for ip, tp in enumerate(["p", "n"]): # hence iband == 0 or 1
        if method=="boltztrap1":
            iband = ip
        else:
            iband = ip*(cbm_idx+nabove_cbm) + (1-ip)*(vbm_idx-nbelow_vbm) + 1
        band , _, _ = interpolate_bs(hs_kpoints, interp_params, iband=iband,
                                      method=method, scissor=scissor,
                                      matrix=bs.structure.lattice.matrix,
                                      n_jobs=n_jobs, sgn=(-1)**ip)
        global_ext_idx = (1-iband) * np.argmax(band) + iband * np.argmin(band)
        if eref is None:
            global_extrema[tp]['energy'] = band[global_ext_idx]
            global_extrema[tp]['kpoint'] = hs_kpoints[global_ext_idx]
        extrema_idx = detect_peaks(band, mph=None, mpd=1,
                               valley=ip==1)

        # making sure CBM & VBM are always included regardless of min_normdiff
        extrema_energies = [band[i] for i in extrema_idx]
        sorted_idx = np.argsort(extrema_energies)
        if tp=='p':
            sorted_idx = sorted_idx[::-1]
        extrema_idx = extrema_idx[sorted_idx]

        extrema_init = []
        for idx in extrema_idx:
            k_localext = hs_kpoints[idx]
            if abs(band[idx] - global_extrema[tp]['energy']) < Ecut[tp]:
                far_enough = True
                for kp in extrema_init:
                    kp = np.array(kp)
                    if norm(to_cart(get_closest_k(k_localext,
                                          np.vstack(
                                              (bs.get_sym_eq_kpoints(-kp),
                                               bs.get_sym_eq_kpoints(kp))),
                                          return_diff=True))) <= min_normdiff:
                        far_enough = False
                if far_enough:
                    extrema_init.append(k_localext)

        # check to see if one of the actual high-symm k-points can be used
        hisymks = list(hsk.kpath['kpoints'].values())
        all_hisymks = []
        for kp in hisymks:
            all_hisymks += list(np.vstack((bs.get_sym_eq_kpoints(-kp),
                                           bs.get_sym_eq_kpoints(kp))))
        for k_ext_found in extrema_init:
            kp = get_closest_k(k_ext_found, all_hisymks, return_diff=False)
            if norm(to_cart(kp - k_ext_found)) < min_normdiff/10.0:
                final_extrema[tp].append(kp)
            else:
                final_extrema[tp].append(k_ext_found)
        # sort the extrema based on their energy (i.e. importance)
        subband, _, _ = interpolate_bs(final_extrema[tp], interp_params, iband=iband,
                                    method=method, scissor=scissor,
                                    matrix=bs.structure.lattice.matrix,
                                    n_jobs=n_jobs, sgn=(-1) ** ip)
        sorted_idx = np.argsort(subband)
        if iband==0:
            sorted_idx = sorted_idx[::-1]
        final_extrema[tp] = [final_extrema[tp][i] for i in sorted_idx]

    if return_global:
        return final_extrema, global_extrema
    else:
        return final_extrema
