import collections
import copy
import logging
from multiprocessing.sharedctypes import RawArray
from typing import Dict, Any

import numpy as np
import scipy
from monty.serialization import dumpfn, loadfn

from amset import amset_defaults
from amset.misc.constants import k_B
from pymatgen import Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.string import latexify_spacegroup

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__date__ = "June 21, 2019"

spin_name = {Spin.up: "spin-up", Spin.down: "spin-down"}

logger = logging.getLogger(__name__)


def create_shared_array(data, return_buffer=False):
    data = np.asarray(data)
    shared_data = RawArray("d", int(np.prod(data.shape)))
    buffered_data = np.frombuffer(shared_data).reshape(data.shape)
    buffered_data[:] = data[:]

    if return_buffer:
        return shared_data, buffered_data
    else:
        return shared_data


def validate_settings(user_settings):
    settings = copy.deepcopy(amset_defaults)

    def recursive_update(d, u):
        """ Recursive dict update."""
        for k, v in u.items():
            if isinstance(v, collections.Mapping):
                d[k] = recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    recursive_update(settings, user_settings)

    # validate the type of some settings
    if isinstance(settings["general"]["doping"], (int, float)):
        settings["general"]["doping"] = [settings["general"]["doping"]]

    elif isinstance(settings["general"]["doping"], str):
        settings["general"]["doping"] = parse_doping(
            settings["general"]["doping"])

    if isinstance(settings["general"]["temperatures"], (int, float)):
        settings["general"]["temperatures"] = [
            settings["general"]["temperatures"]]
    elif isinstance(settings["general"]["temperatures"], str):
        settings["general"]["temperatures"] = parse_temperatures(
            settings["general"]["temperatures"])

    if isinstance(settings["material"]["deformation_potential"], str):
        settings["general"]["deformation_potential"] = \
            parse_deformation_potential(
                settings["general"]["deformation_potential"])

    settings["general"]["doping"] = np.asarray(settings["general"]["doping"])
    settings["general"]["temperatures"] = np.asarray(
        settings["general"]["temperatures"])

    return settings


def tensor_average(tensor):
    return np.average(scipy.linalg.eigvalsh(tensor))


def groupby(a, b):
    # Get argsort indices, to be used to sort a and b in the next steps
    sidx = b.argsort(kind='mergesort')
    a_sorted = a[sidx]
    b_sorted = b[sidx]

    # Get the group limit indices (start, stop of groups)
    cut_idx = np.flatnonzero(
        np.r_[True, b_sorted[1:] != b_sorted[:-1], True])

    # Split input array with those start, stop ones
    out = np.array(
        [a_sorted[i:j] for i, j in zip(cut_idx[:-1], cut_idx[1:])])
    return out


def unicodeify_spacegroup(spacegroup_symbol: str):
    subscript_unicode_map = {
        0: "₀",
        1: "₁",
        2: "₂",
        3: "₃",
        4: "₄",
        5: "₅",
        6: "₆",
        7: "₇",
        8: "₈",
        9: "₉",
    }

    symbol = latexify_spacegroup(spacegroup_symbol)

    for number, unicode_number in subscript_unicode_map.items():
        symbol = symbol.replace("$_{" + str(number) + "}$", unicode_number)

    overline = "\u0305"  # u"\u0304" (macron) is also an option

    symbol = symbol.replace("$\\overline{", overline)
    symbol = symbol.replace("$", "")
    symbol = symbol.replace("{", "")
    symbol = symbol.replace("}", "")

    return symbol


def write_settings_to_file(settings: Dict[str, Any], filename: str):
    """Write amset configuration settings to a formatted yaml file.

    Args:
        settings: The configuration settings.
        filename: A filename.
    """
    settings = cast_dict(settings)
    dumpfn(settings, filename, indent=4, default_flow_style=False)


def load_settings_from_file(filename: str) -> Dict[str, Any]:
    """Load amset configuration settings from a yaml file.

    If the settings file does not contain a required parameter, the default
    value will be added to the configuration.

    An example file is given in *amset/examples/example_settings.yaml*.

    Args:
        filename: Path to settings file.

    Returns:
        The settings, with any missing values set according to the amset
        defaults.
    """
    logger.info("Loading settings from: {}".format(filename))
    settings = loadfn(filename)

    return validate_settings(settings)


def cast_dict(d):
    new_d = {}
    for k, v in d.items():
        # cast keys
        if isinstance(k, Spin):
            k = k.value

        if isinstance(v, collections.Mapping):
            new_d[k] = cast_dict(v)
        else:
            # cast values
            if isinstance(v, np.ndarray):
                v = v.tolist()
            elif isinstance(v, tuple):
                v = list(v)

            new_d[k] = v
    return new_d


def gen_even_slices(n, n_packs):
    """Generator to create n_packs slices going up to n.

    Parameters
    ----------
    n : int
    n_packs : int
        Number of slices to generate.

    Yields
    ------
    slice

    Examples
    --------
    """
    start = 0
    if n_packs < 1:
        raise ValueError("gen_even_slices got n_packs=%s, must be >=1"
                         % n_packs)
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            yield slice(start, end, None)
            start = end


def kpoints_to_first_bz(kpoints: np.ndarray) -> np.ndarray:
    """Translate fractional k-points to the first Brillouin zone.

    I.e. all k-points will have fractional coordinates:
        -0.5 <= fractional coordinates < 0.5

    Args:
        kpoints: The k-points in fractional coordinates.

    Returns:
        The translated k-points.
    """
    kp = kpoints - np.round(kpoints)
    kp[kp == 0.5] = -0.5
    return kp


def parse_doping(doping_str: str):
    doping_str = doping_str.strip().replace(" ", "")

    try:
        if ":" in doping_str:
            parts = list(map(float, doping_str.split(":")))

            if len(parts) != 3:
                raise ValueError

            return np.geomspace(parts[0], parts[1], int(parts[2]))

        else:
            return np.array(list(map(float, doping_str.split(","))))

    except ValueError:
        raise ValueError("ERROR: Unrecognised doping format: {}".format(
            doping_str))


def parse_temperatures(temperatures_str: str):
    temperatures_str = temperatures_str.strip().replace(" ", "")

    try:
        if ":" in temperatures_str:
            parts = list(map(float, temperatures_str.split(":")))

            if len(parts) != 3:
                raise ValueError

            return np.linspace(parts[0], parts[1], int(parts[2]))

        else:
            return np.array(list(map(float, temperatures_str.split(","))))

    except ValueError:
        raise ValueError("ERROR: Unrecognised temperature format: {}".format(
            temperatures_str))


def parse_deformation_potential(deformation_pot_str: str):
    deformation_pot_str = deformation_pot_str.strip().replace(" ", "")

    try:
        parts = list(map(float, deformation_pot_str.split(",")))

        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return tuple(parts)
        else:
            raise ValueError

    except ValueError:
        raise ValueError("ERROR: Unrecognised deformation potential format: "
                         "{}".format(deformation_pot_str))


def f0(energy, fermi, temperature):
    """
    Returns the value of Fermi-Dirac distribution at equilibrium.

    Args:
        energy (float): energy in eV
        fermi (float): the Fermi level with the same reference as E (in eV)
        temperature (float): the absolute temperature in Kelvin.

    Returns (0<float<1):
        The occupation calculated by Fermi dirac
    """
    return 1. / (1. + np.exp((energy - fermi) / (k_B * temperature)))


def df0de(energy, fermi, temperature):
    """
    Returns the energy derivative of the Fermi-Dirac equilibrium distribution

    Args: see Args for f0(energy, fermi, temperature)

    Returns (float<0): the energy derivative of the Fermi-Dirac distribution.
    """
    exponent = (energy - fermi) / (k_B * temperature)
    result = -1 / (k_B * temperature) * \
           np.exp((energy - fermi) / (k_B * temperature)) / (
                   1 + np.exp((energy - fermi) / (k_B * temperature))) ** 2
    # This is necessary so at too low numbers python doesn't return NaN
    result[(exponent > 40) | (exponent < -40)] = 1e-32
    return result


def get_dense_kpoint_mesh_spglib(mesh, spg_order=False, shift=0):
    """This is a reimplementation of the spglib c function get_all_grid_addresses

    Given a k-point mesh, gives the full k-point mesh that covers
    the first Brillouin zone. Uses the same convention as spglib,
    in that k-points on the edge of the Brillouin zone only
    appear as positive numbers. I.e. coordinates will always be
    +0.5 rather than -0.5. Similarly, the same ordering scheme is
    used.

    The only difference between this function and the function implemented
    in spglib is that here we return the final fraction coordinates
    whereas spglib returns the grid_addresses (integer numbers).
    """
    mesh = np.asarray(mesh)

    addresses = np.stack(np.mgrid[0:mesh[0], 0:mesh[1], 0:mesh[2]],
                         axis=-1).reshape(np.product(mesh), -1)

    if spg_order:
        # order the kpoints using the same ordering scheme as spglib
        idx = addresses[:, 2] * (mesh[0] * mesh[1]) + addresses[:, 1] * mesh[
            0] + addresses[:, 0]
        addresses = addresses[idx]

    addresses -= mesh * (addresses > mesh / 2)
    # return (addresses + shift) / mesh

    full_kpoints = (addresses + shift) / mesh
    sort_idx = np.lexsort((full_kpoints[:, 2], full_kpoints[:, 2] < 0,
                           full_kpoints[:, 1], full_kpoints[:, 1] < 0,
                           full_kpoints[:, 0], full_kpoints[:, 0] < 0))
    full_kpoints = full_kpoints[sort_idx]
    return full_kpoints


def get_dense_kpoint_mesh(mesh):
    kpts = np.stack(
        np.mgrid[
        0:mesh[0] + 1,
        0:mesh[1] + 1,
        0:mesh[2] + 1],
        axis=-1).reshape(-1, 3).astype(float)

    # remove central point for all even ndim as this will fall
    # exactly the in the centre of the grid and will be on top of
    # the original k-point
    if not any(mesh % 2):
        kpts = np.delete(kpts, int(1 + np.product(mesh + 1) / 2), axis=0)

    kpts /= mesh  # gets frac kpts between 0 and 1
    kpts -= 0.5
    return kpts


def get_symmetry_equivalent_kpoints(structure, kpoints, symprec=0.1, tol=1e-6,
                                    return_inverse=False,
                                    rotation_matrices=None,
                                    time_reversal_symmetry=True):
    round_dp = int(np.log10(1 / tol))

    def shift_and_round(k):
        k = kpoints_to_first_bz(k)
        k = np.round(k, round_dp)
        return list(map(tuple, k))

    kpoints = np.asarray(kpoints)
    round_kpoints = shift_and_round(kpoints)

    if rotation_matrices is None:
        sg = SpacegroupAnalyzer(structure, symprec=symprec)
        symmops = sg.get_symmetry_operations(cartesian=False)
        rotation_matrices = np.array([o.rotation_matrix for o in symmops])

    if time_reversal_symmetry:
        # TODO: Remove equivalent rotation matrices
        rotation_matrices = np.concatenate(
            (rotation_matrices, -rotation_matrices))

    cart_rotation_matrices = np.array(
        [similarity_transformation(
            structure.lattice.reciprocal_lattice.matrix, r.T)
         for r in rotation_matrices])

    equiv_points_mapping = {}
    rotation_matrix_mapping = {}
    mapping = []
    rot_mapping = []

    for i, point in enumerate(round_kpoints):

        if point in equiv_points_mapping:
            map_idx = equiv_points_mapping[point]
            mapping.append(map_idx)
            rot_mapping.append(rotation_matrix_mapping[map_idx][point])
        else:
            new_points = shift_and_round(np.dot(kpoints[i], rotation_matrices))

            equiv_points_mapping.update(zip(new_points, [i] * len(new_points)))
            rotation_matrix_mapping[i] = dict(
                zip(new_points, cart_rotation_matrices))

            mapping.append(i)
            rot_mapping.append(np.eye(3))

    ir_kpoints_idx, ir_to_full_idx, weights = np.unique(
        mapping, return_inverse=True, return_counts=True)

    ir_kpoints = kpoints[ir_kpoints_idx]

    if return_inverse:
        return (ir_kpoints, weights, ir_kpoints_idx, ir_to_full_idx,
                np.array(mapping), np.array(rot_mapping))
    else:
        return ir_kpoints, weights


def similarity_transformation(rot, mat):
    """ R x M x R^-1 """
    return np.dot(rot, np.dot(mat, np.linalg.inv(rot)))


class SymmetryEquivalizer(object):

    def __init__(self, structure, symprec=0.1, tol=1e-6,
                 time_reversal_symmetry=True):
        self.round_dp = int(np.log10(1 / tol))
        sg = SpacegroupAnalyzer(structure, symprec=symprec)
        symmops = sg.get_symmetry_operations(cartesian=False)
        self.rotation_matrices = np.array([o.rotation_matrix for o in symmops])

        if time_reversal_symmetry:
            # TODO: Remove equivalent rotation matrices
            self.rotation_matrices = np.concatenate(
                (self.rotation_matrices, -self.rotation_matrices))

        self.cart_rotation_matrices = np.array(
            [similarity_transformation(
                structure.lattice.reciprocal_lattice.matrix, r.T)
                for r in self.rotation_matrices])

        self.equiv_points_mapping = {}
        self.rotation_matrix_mapping = {}
        self.index = 0

    def get_equivalent_kpoints(self, kpoints):

        def shift_and_round(k):
            k = kpoints_to_first_bz(k)
            k = np.round(k, self.round_dp)
            return list(map(tuple, k))

        kpoints = np.asarray(kpoints)
        round_kpoints = shift_and_round(kpoints)

        mapping = []
        rot_mapping = []

        for i, point in enumerate(round_kpoints):

            if point in self.equiv_points_mapping:
                map_idx = self.equiv_points_mapping[point]
                mapping.append(map_idx)
            else:
                map_idx = i + self.index
                new_points = shift_and_round(
                    np.dot(kpoints[i], self.rotation_matrices))
                self.equiv_points_mapping.update(
                    zip(new_points, [map_idx] * len(new_points)))

                mapping.append(map_idx)
                rot_mapping.append(np.eye(3))

        self.index += len(round_kpoints)

        _, ir_kpoints_idx, ir_to_full_idx = np.unique(
            mapping, return_inverse=True, return_index=True)

        return ir_kpoints_idx, ir_to_full_idx, np.array(mapping)
