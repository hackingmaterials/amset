import logging

import numpy as np

from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional, Union, Tuple, List, Sequence, Any, Dict

from BoltzTraP2.bandlib import DOS

from monty.json import MSONable
from scipy.ndimage import gaussian_filter1d
from spglib import spglib

from amset.logging import LoggableMixin
from amset.utils.band_structure import kpoints_to_first_bz, get_closest_k
from amset.utils.constants import k_B, A_to_nm
from amset.utils.detect_peaks import detect_peaks
from amset.utils.general import norm

from pymatgen import Spin
from pymatgen.electronic_structure.bandstructure import BandStructure, \
    BandStructureSymmLine
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.coord import pbc_diff


class AbstractInterpolater(MSONable, LoggableMixin, ABC):

    def __init__(self, band_structure: BandStructure, num_electrons: int,
                 calc_dir: str = '.', logger: Union[bool, Logger] = True,
                 log_level: int = logging.INFO, symprec: float = 0.01,
                 soc: bool = False):
        self._band_structure = band_structure
        self._num_electrons = num_electrons
        self._calc_dir = calc_dir
        self._logger = self.get_logger(logger, level=log_level)
        self._symprec = symprec
        self._soc = soc
        self._lattice_matrix = band_structure.structure.lattice.matrix
        self._sga = SpacegroupAnalyzer(band_structure.structure,
                                       symprec=symprec)
        self._vbm_idx = max(self._band_structure.get_vbm()
                            ['band_index'][Spin.up])
        self._offset = 0.
        self.initialize()

    def initialize(self):
        """Initialise the interpolater."""
        vbm_kpt = self._band_structure.get_vbm()['kpoint'].frac_coords
        vbm_e = self._band_structure.get_vbm()['energy']
        self._offset += vbm_e - self.get_energies([vbm_kpt], self._vbm_idx)[0]

    @abstractmethod
    def get_energies(self, kpoints: Union[np.ndarray, List],
                     iband: Optional[Union[int, List[int]]] = None,
                     scissor: float = 0.0,
                     return_velocity: bool = False,
                     return_effective_mass: bool = False,
                     coords_are_cartesian: bool = False,
                     ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Gets the interpolated energies for multiple k-points in a band.

        Args:
            kpoints: The k-point coordinates.
            iband: A band index or list of band indicies for which to get the
                energies. Band indices are 0-indexed. If ``None``, the energies
                for all available bands will be returned.
            scissor: The amount by which the band gap is scissored.
            return_velocity: Whether to return the band velocities.
            return_effective_mass: Whether to return the band effective masses.
            coords_are_cartesian: Whether the kpoints are in cartesian or
                fractional coordinates.

        Returns:
            The band energies as a numpy array. If ``return_velocity`` or
            ``return_effective_mass`` are ``True`` a tuple is returned,
            formatted as::

                (energies, Optional[velocities], Optional[effective_masses])

            The velocities and effective masses are given as the 1x3 trace and
            full 3x3 tensor, respectively (along cartesian directions).
        """
        pass

    def get_dos(self, kpoint_mesh: List[int],
                iband: Optional[Union[int, List[int]]] = None,
                emin: float = None, emax: float = None, estep: float = 0.001,
                width: float = 0.05, scissor: float = 0.0,
                integral_sphere: Optional[Tuple[List[float], float]] = None
                ) -> np.ndarray:
        """Calculates the density of states using the interpolated bands.

        # TODO: tetrahedron method interpolation for better DOS at coarser kgrid

        Args:
            kpoint_mesh: The k-point mesh as a 1x3 array. E.g.,``[6, 6, 6]``.
            iband: A band index or list of band indicies for which to calculate
                the DOS. Band indices are 0-indexed. If ``None``, the energies
                for all available bands will be returned.
            emin: The minium energy. If ``None`` it will be calculated
                automatically from the band energies.
            emax: The maximum energy. If ``None`` it will be calculated
                automatically from the band energies.
            estep: The energy step, where smaller numbers give more
                accuracy but are more expensive.
            width: The gaussian smearing width.
            scissor: The amount by which the band gap is scissored.
            integral_sphere: Whether to limit the calculation of DOS to a
                particular region of k-space. The sphere is defined by
                its center in fractional coordinates and the radius in
                1/nm. For example::

                    ([0.5, 0.5, 0.5], 5)

                Defines a sphere centered at [0.5, 0.5, 0.5] with a radius of
                5 nm$^-1$.

        Returns:
            The density of states data, formatted as::

                (energies, densities)
        """

        if integral_sphere:
            # get full k-point mesh (not IR reduced)
            _, grid = spglib.get_ir_reciprocal_mesh(
                kpoint_mesh, self._sga._cell, self._symprec)
            mesh_data = grid / kpoint_mesh

            # calculate the distances in nm of each k-point and sphere centre
            diff_frac = (
                self._band_structure.structure.lattice.reciprocal_lattice.
                get_cartesian_coords(pbc_diff(mesh_data, integral_sphere[0])))
            k_distance_nm = 1 / A_to_nm * np.linalg.norm(diff_frac, axis=1)

            # only include k-points within distance tolerance
            kpoints = [k for k, d in zip(mesh_data, k_distance_nm)
                       if d <= integral_sphere[1]]
            weights = np.ones(len(kpoints), dtype=int)
            normalization_factor = len(kpoints) / len(mesh_data)

        else:
            mesh_data = np.array(self._sga.get_ir_reciprocal_mesh(kpoint_mesh))
            kpoints = np.asarray(list(map(list, mesh_data[:, 0])))
            weights = mesh_data[:, 1]
            normalization_factor = 1

        if isinstance(iband, int):
            iband = [iband]

        energies = np.array(self.get_energies(kpoints, scissor=scissor,
                                              iband=iband))

        nbands = energies.shape[0]
        nkpts = energies.shape[1]
        emin = emin if emin else np.min(energies) - width * 5
        emax = emax if emax else np.max(energies) + width * 5
        epoints = int(round((emax - emin) / estep))

        # BoltzTraP DOS weights don't work so include the degeneracy manually
        all_energies = np.array([[energies[nb][nk] for nb in range(nbands)]
                                 for nk in range(nkpts)
                                 for _ in range(weights[nk])])

        emesh, dos = DOS(all_energies, erange=(emin, emax), npts=epoints)
        dos = gaussian_filter1d(dos, width / (emesh[1] - emesh[0]))

        normalization_factor *= 1 if self._soc else 2
        dos *= normalization_factor

        self.logger.debug("dos normalization factor: {}".format(
            normalization_factor))

        return np.column_stack((emesh, dos))

    def get_extrema(self, iband: int, line_density: int = 30,
                    min_norm_diff: float = 4.0, e_cut: float = 10 * k_B * 300,
                    return_global_extrema: bool = False, scissor: float = 0.0
                    ) -> Union[List[int], Tuple[List[int], Dict[str, Any]]]:
        """Gets band extrema by exploring the high-symmetry k-point path.

        If the band is a valence band, maxima will be returned. For conduction
        bands, minima will be returned.

        Args:
            iband: A band index for which to get the extrema. Band indices are
                0-indexed.
            line_density: The maximum number of k-points between each two
                consecutive high-symmetry k-points
            min_norm_diff: The minimum allowed distance (in 1/nm) between
                extrema. Required to avoid numerical instability errors
                and finding peaks that are too close to each other for the Amset
                formulation to be relevant.
            e_cut: The maximum energy cut-off from the global extrema point.
            return_global_extrema: Whether to return the global extrema k-point
                and energy.
            scissor: The amount by which the band gap is scissored.

        Returns:
            The band extrema as a list of k-points. If ``return_global_extrema``
            is ``True``, the data will be returned as a tuple of::

                (extrema, global_extrema)

            Where ``global_extrema`` is a dictionary with the keys ``"k-point"``
            and ``"energy"``.
        """

        # TODO: Rewrite this to accept multiple bands
        is_cb = iband > self._vbm_idx

        def kpoints_are_separated(a_kpoint, list_kpoints):
            sym_kpoints = [k for lk in list_kpoints for dk in (lk, -lk)
                           for k in self._band_structure.get_sym_eq_kpoints(dk)]
            if sym_kpoints:
                return norm(self._to_cart(get_closest_k(
                    a_kpoint, sym_kpoints, return_diff=True))) > min_norm_diff
            else:
                return True

        hsk = HighSymmKpath(self._band_structure.structure)
        kpoints = kpoints_to_first_bz(hsk.get_kpoints(
            line_density=line_density, coords_are_cartesian=False)[0])
        band = self.get_energies(kpoints, iband=iband,
                                 scissor=scissor)

        global_extrema_idx = np.argmin(band) if is_cb else np.argmax(band)
        global_extrema = {'energy': band[global_extrema_idx],
                          'kpoint': kpoints[global_extrema_idx]}

        # order the extrema by energy to ensure VBM/CBM always included
        extrema_idx = detect_peaks(band, mph=None, mpd=1, valley=is_cb)
        extrema_idx = sorted(extrema_idx,
                             key=lambda x: band[x], reverse=not is_cb)

        # filter those too far away in energy
        extrema_idx = [x for x in extrema_idx
                       if abs(band[x] - global_extrema['energy']) < e_cut]

        # filter to only include extrema separated in k-space
        extrema = []
        for kpoint_idx in extrema_idx:
            if kpoints_are_separated(kpoints[kpoint_idx], extrema):
                extrema.append(kpoints[kpoint_idx])
        extrema = np.array(extrema)

        # sort the extrema based on their energy (i.e. importance)
        band = self.get_energies(extrema, iband=iband, scissor=scissor)
        sorted_idx = np.argsort(band) if is_cb else np.argsort(band)[::-1]
        extrema = extrema[sorted_idx]

        if return_global_extrema:
            return extrema, global_extrema
        else:
            return extrema

    def get_line_mode_band_structure(
            self, line_density: int = 50, scissor: float = 0.,
            iband: Optional[Union[int, List[int]]] = None
            ) -> BandStructureSymmLine:
        """Gets the interpolated band structure along high symmetry directions.

        Args:
            line_density: The maximum number of k-points between each two
                consecutive high-symmetry k-points
            scissor: The amount by which the band gap is scissored.
            iband: A band index or list of band indicies for which to calculate
                the DOS. Band indices are 0-indexed. If ``None``, the energies
                for all available bands will be returned.

        Returns:
            The line mode band structure.
        """
        if isinstance(iband, int):
            iband = [iband]

        hsk = HighSymmKpath(self._band_structure.structure,
                            symprec=self._symprec)
        kpoints, labels = hsk.get_kpoints(line_density=line_density,
                                          coords_are_cartesian=True)
        labels_dict = {label: kpoint for kpoint, label
                       in zip(kpoints, labels) if label != ''}

        energies = self.get_energies(kpoints, iband=iband, scissor=scissor,
                                     coords_are_cartesian=True)
        return BandStructureSymmLine(kpoints, {Spin.up: energies},
                                     self._band_structure.structure.lattice,
                                     self._band_structure.efermi,
                                     labels_dict,
                                     coords_are_cartesian=True)

    @staticmethod
    def _simplify_return_data(data: Sequence[Any],
                              ) -> Union[Any, Sequence[Any]]:
        """Helper function to prepare data to be returned.

        Takes a collection of objects and:

        - If the collection contains only a single item, return just that.
        - Otherwise return the full collection.

        Args:
            data: The data.

        Returns:
            The first element of the collection if it only has one element, else
            the full collection.
        """

        if len(data) == 1:
            return data[0]
        else:
            return tuple(data)

    def _to_cart(self, k):
        """Convert fractional k-points to cartesian coordinates in 1/nm."""
        return np.dot(self._band_structure.structure.lattice.
                      reciprocal_lattice.matrix, k) * 10
