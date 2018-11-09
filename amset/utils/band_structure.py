import numpy as np

from amset.utils.general import AmsetError, norm
from pymatgen import Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import pbc_diff

"""
For functions that deal with band structure, k-points operations, k-point grid, 
Brillouin zone and even parsing s- and p-orbital contribution from vasprun.xml
"""


def kpts_to_first_BZ(kpts):
    """
    Brings a list of k-points to the 1st Brillouin Zone (BZ);
    i.e. -0.5 <= the fractional coordinates <= 0.5

    Args:
        kpts ([3x1 list or array]): list of k-points fractional coordinates

    Returns ([3x1 list or array]): list of transformed coordinates
    """
    new_kpts = []
    for i, k in enumerate(kpts):
        for alpha in range(3):
            while k[alpha] > 0.50:
                k[alpha] -= 1.00
            while k[alpha] < -0.50:
                k[alpha] += 1.00
        new_kpts.append(k)
    return new_kpts


def get_closest_k(kpoint, ref_ks, return_diff=False, exclude_self=False):
    """
    Returns the list of difference between kpoints. If return_diff True, then
        for a given kpoint the minimum distance among distances with ref_ks is
        returned or just the reference kpoint that results if not return_diff

    Args:
        kpoint (1x3 array): the coordinates of the input k-point
        ref_ks ([1x3 array]): list of reference k-points from which the
            distance with initial_ks are calculated
        return_diff (bool): if True, the minimum distance is returned
        exclude_self (bool): if kpoint is already repeated in ref_ks, exclude
            that from ref_ks

    Returns (1x3 array):
    """
    if len(list(kpoint)) != 3 or len(list(ref_ks[0])) != 3:
        raise AmsetError('k-point coordinates must be 3-dimensional')
    norms = [norm(ki-kpoint) for ki in ref_ks]
    if exclude_self:
        norms = [norm if norm>0.001 else 1e10 for norm in norms]
    min_dist_ik = np.array(norms).argmin()
    if return_diff:
        return kpoint - ref_ks[min_dist_ik]
    else:
        return ref_ks[min_dist_ik]


def remove_duplicate_kpoints(kpts, dk=0.01, periodic=True):
    """
    Removes duplicate points from a list of k-points.

    Args:
        kpts ([np.ndarray or list]): list of k-point coordinates

    Returns ([np.ndarray or list]):
        kpts but with duplicate points removed.
    """
    rm_list = []
    if periodic:
        diff_func = pbc_diff
    else:
        diff_func = np.subtract
    for i in range(len(kpts) - 1):
        for j in range(i + 1, len(kpts)):
            if np.allclose(diff_func(kpts[i], kpts[j]), [0, 0, 0], atol=dk):
                rm_list.append(i)
                break
    return [list(k) for k in np.delete(kpts, rm_list, axis=0)]


def get_bindex_bspin(extremum, is_cbm):
    """
    Returns the band index and spin of band extremum

    Args:
        extremum (dict): dictionary containing the CBM/VBM, i.e. output of
            Bandstructure.get_cbm()
        is_cbm (bool): whether the extremum is the CBM or not
    """

    idx = int(is_cbm) - 1  # 0 for CBM and -1 for VBM
    try:
        bidx = extremum["band_index"][Spin.up][idx]
        bspin = Spin.up
    except IndexError:
        bidx = extremum["band_index"][Spin.down][idx]
        bspin = Spin.down
    return bidx, bspin


def insert_intermediate_kpoints(kpts, n=2):
    """
    Insert n k-points in between each two k-points from kpts and return the
        latter bigger list. This can be used for example to make a finer-mesh
        HighSymmKpath

    Args:
        kpts ([[float]]): list of coordinates of k-points
        n (int): the number of k-points inserted between each pair of k-points

    Returns ([[float]]): the final list of k-point coordinates
    """
    n += 1
    new_kpts = []
    for i in range(len(kpts)-1):
        step = (kpts[i+1] - kpts[i])/n
        for j in range(n):
            new_kpts.append(kpts[i] + j*step)
    new_kpts.append(kpts[-1])
    return new_kpts


def get_dft_orbitals(vasprun, bidx, lorbit):
    """
    The contribution from s and p orbitals at a given band for kpoints
    that were used in the DFT run (from which vasprun.xml is read). This is
    just to parse the total orbital contributions out of Vasprun depending
    on LORBIT.

    Args:
        vasprun (pymatgen Vasprun):
        bidx (idx): band index
        lorbit (int): the LORBIT flag that was used when vasprun.xml generated

    Returns:
        ([float], [float]) two lists: s&p orbital scores at the band # bidx
    """
    projected = vasprun.projected_eigenvalues
    nk = len(vasprun.actual_kpoints)
    # projected indexes : Spin; kidx; bidx; s,py,pz,px,dxy,dyz,dz2,dxz,dx2
    s_orbital = [0.0] * nk
    p_orbital = [0.0] * nk
    for ik in range(nk):
        s_orbital[ik] = sum(projected[Spin.up][ik][bidx])[0]
        if lorbit == 10:
            p_orbital[ik] = sum(projected[Spin.up][ik][bidx])[1]
        elif lorbit == 11:
            p_orbital[ik] = sum(sum(projected[Spin.up][ik][bidx])[1:4])
        else:
            raise AmsetError('Not sure what to do with lorbit={}'.format(lorbit))
    return s_orbital, p_orbital


def generate_adaptive_kmesh(bs, important_points, kgrid_tp, ibz=True):
    """
    Returns a kpoint mesh surrounding the important k-points in the
    conduction (n-type) and valence bands (p-type). This mesh is adaptive,
    meaning that the mesh is much finer closer to these "important" points
    than it is further away. This saves tremendous computational time as
    the points closer to the extremum are the most important ones dictating
    the transport properties.

    Args:
        bs (pymatgen.BandStructure): the bandstructure with bs.structure
            present that are used for structural symmetry and getting the
            symmetrically equivalent points
        important_points ({"n": [[3x1 array]], "p": [[3x1 array]]}): list
            of fractional coordinates of extrema for n-type and p-type
        kgrid_tp (str): determines how coarse/fine the k-mesh would be.
            options: "very coarse", "coarse", "fine", "very fine"
        ibz (bool): whether to generate the k-mesh based on a scaled k-mesh
            in Irreducible Brillouin Zone (True, recommended) or custom
            intervals only in the directions (+/-) of ibz kpoints.

    Returns ({"n": [[3x1 array]], "p": [[3x1 array]]}):
        list of k-points (k-mesh) may be different for conduction or
        valence bands.
    """
    if ibz:
        kpts = {}
        kgrid_tp_map = {'very coarse': 4,
                        'coarse': 8,
                        'fine': 14,
                        'very fine': 19,
                        'super fine': 25,
                        'extremely fine': 33
                        }
        nkk = kgrid_tp_map[kgrid_tp]
        sg = SpacegroupAnalyzer(bs.structure)
        kpts_and_weights = sg.get_ir_reciprocal_mesh(mesh=(nkk, nkk, nkk),
                                                     is_shift=[0, 0, 0])
        kpts_and_weights_init = sg.get_ir_reciprocal_mesh(mesh=(5, 5, 5),
                                                     is_shift=[0, 0, 0])
        initial_ibzkpt_init = [i[0] for i in kpts_and_weights_init]
        initial_ibzkpt0 = np.array([i[0] for i in kpts_and_weights])

        # this is to cover the whole BZ in an adaptive manner in case of high-mass isolated valleys
        initial_ibzkpt0 = np.array(initial_ibzkpt_init + \
                                   list(initial_ibzkpt0/3.0) +
                                   list(initial_ibzkpt0/9.0) + \
                                   list(initial_ibzkpt0/19.0) )
        for tp in ['p', 'n']:
            tmp_kpts = []
            for important_point in important_points[tp]:
                initial_ibzkpt = initial_ibzkpt0 + important_point
                for k in initial_ibzkpt:
                    tmp_kpts += list(bs.get_sym_eq_kpoints(k))
            kpts[tp] = tmp_kpts
    else:
        if kgrid_tp == "fine":
            mesh = np.array(
                [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.25])
            nkk = 15
        elif kgrid_tp == "coarse":
            mesh = np.array([0.001, 0.005, 0.03, 0.1])
            nkk = 10
        elif kgrid_tp == 'very coarse':
            mesh = np.array([0.001, 0.01])
            nkk = 5
        else:
            raise AmsetError('Unsupported kgrid_tp: {}'.format(kgrid_tp))
        # just to find a set of + or - kpoint signs when ibzkpt is generated:
        sg = SpacegroupAnalyzer(bs.structure)
        kpts_and_weights = sg.get_ir_reciprocal_mesh(mesh=(nkk, nkk, nkk),
                                                     is_shift=[0, 0, 0])
        initial_ibzkpt = [i[0] for i in kpts_and_weights]
        step_signs = [[np.sign(k[0]), np.sign(k[1]), np.sign(k[2])] for k in
                      initial_ibzkpt]
        step_signs = remove_duplicate_kpoints(step_signs, periodic=False)

        # actually generating the mesh seaprately for n- and p-type
        kpts = {'n': [], 'p': []}
        for tp in ["n", "p"]:
            for k_extremum in important_points[tp]:
                for kx_sign, ky_sign, kz_sign in step_signs:
                    for kx in k_extremum[0] + kx_sign*mesh:
                        for ky in k_extremum[1] + ky_sign*mesh:
                            for kz in k_extremum[2] + kz_sign*mesh:
                                kpts[tp].append(np.array([kx, ky, kz]))
            tmp_kpts = []
            for k in kpts[tp]:
                tmp_kpts += list(bs.get_sym_eq_kpoints(k))
            kpts[tp] = remove_duplicate_kpoints(tmp_kpts, dk=0.0009)
    return kpts
