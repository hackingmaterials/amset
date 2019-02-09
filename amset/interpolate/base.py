import logging

import numpy as np

from abc import ABC, abstractmethod
from logging import Logger
from typing import Optional, Union, Tuple, List, Sequence, Any
from scipy.integrate import trapz

from monty.json import MSONable

from amset.logging import LoggableMixin
from amset.utils.band_structure import kpts_to_first_bz, get_closest_k
from amset.utils.constants import k_B
from amset.utils.detect_peaks import detect_peaks
from amset.utils.general import norm
from pymatgen import Spin
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath


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

    @abstractmethod
    def get_energies(self, kpoints: Union[np.ndarray, List],
                     iband: Optional[Union[int, List[int]]] = None,
                     scissor: float = 0.0,
                     return_velocity: bool = False,
                     return_effective_mass: bool = False
                     ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        """Gets the interpolated energies for multiple k-points in a band.

        Args:
            kpoints: The k-points in fractional coordinates.
            iband: A band index or list of band indicies for which to get the
                energies. Band indices are 0-indexed. If ``None``, the energies
                for all available bands will be returned.
            scissor: The amount by which the band gap is scissored.
            return_velocity: Whether to return the band velocities.
            return_effective_mass: Whether to return the band effective masses.

        Returns:
            The band energies as a numpy array. If ``return_velocity`` or
            ``return_effective_mass`` are ``True`` a tuple is returned,
            formatted as::

                (energies, Optional[velocities], Optional[effective_masses])

            The velocities and effective masses are given as the 1x3 trace and
            full 3x3 tensor, respectively (along cartesian directions).
        """
        pass

    def get_dos(self, kpoint_mesh: List[int], emin: float = None,
                emax: float = None, estep: float = 0.001,
                width: float = 0.2, scissor: float = 0.0,
                normalize: bool = False) -> np.ndarray:
        """Calculates the density of states using the interpolated bands.

        Args:
            kpoint_mesh: The k-point mesh as a 1x3 array. E.g.,``[6, 6, 6]``.
            emin: The minium energy. If ``None`` it will be calculated
                automatically from the band energies.
            emax: The maximum energy. If ``None`` it will be calculated
                automatically from the band energies.
            estep: The energy step, where smaller numbers give more
                accuracy but are more expensive.
            width: The gaussian smearing width.
            scissor: The amount by which the band gap is scissored.
            normalize: Whether to normalize the DOS.

        Returns:
            The density of states data, formatted as::

                (energies, densities)
        """
        mesh_data = np.array(self._sga.get_ir_reciprocal_mesh(kpoint_mesh))
        ir_kpts = np.asarray(list(map(list, mesh_data[:, 0])))
        weights = mesh_data[:, 1] / mesh_data[:, 1].sum()

        energies = self.get_energies(ir_kpts, scissor=scissor)
        nbands = energies.shape[0]

        emin = emin if emin else np.min(energies)
        emax = emax if emax else np.max(energies)

        height = 1.0 / (width * np.sqrt(2 * np.pi))
        epoints = int(round((emax - emin) / estep))
        emesh = np.linspace(emin, emax, num=epoints, endpoint=True)
        dos = np.zeros(len(emesh))

        for ik, w in enumerate(weights):
            for b in range(nbands):
                g = height * np.exp(
                    -((emesh - energies[b, ik]) / width) ** 2 / 2.)
                dos += w * g

        if normalize:
            # taken from old amset. Might reintroduce if needed to pass tests
            # for idx in range(self.cbm_dos_idx, self.get_Eidx_in_dos(
            #         self.cbm_vbm["n"]["energy"] + 2.0)):
            #     dos[idx] = max(dos[idx], free_e_dos(
            #         energy=self.dos_emesh[idx] - self.cbm_vbm["n"]["energy"]))
            # for idx in range(
            #         self.get_Eidx_in_dos(self.cbm_vbm["p"]["energy"] - 2.0),
            #         self.vbm_dos_idx):
            #     dos[idx] = max(dos[idx], free_e_dos(
            #         energy=self.cbm_vbm["p"]["energy"] - self.dos_emesh[idx]))

            normalization_factor = nbands * (1 if self._soc else 2)
            integ = trapz(dos, x=emesh)
            self.logger.info("dos integral from {:.3f} to {:.3f} "
                             "eV: {:.3f}".format(emin, emax, integ))
            dos /= integ * normalization_factor

        return np.column_stack((emesh, dos))

    def get_extrema(self, iband, line_density=30, min_normdiff=4.0,
                    e_cut=10 * k_B * 300, return_global=False, scissor=0.0):
        """
        Returns a dictionary of p-type (valence) and n-type (conduction) band
            extrema k-points by looking at the 1st and 2nd derivatives of the
            bands

        Args:
            iband: A band index for which to get the extrema. Band indices are
                0-indexed.
            line_density (int): maximum number of k-points between each two
                consecutive high-symmetry k-points
            min_normdiff (float): the minimum allowed distance
                norm(cartesian k in 1/nm) in extrema; this is important to avoid
                numerical instability errors or finding peaks that are too close
                to each other for Amset formulation to be relevant.
            e_cut (float or dict): max energy difference with CBM/VBM allowed
                for extrema. Valid examples: 0.25 or {'n': 0.5, 'p': 0.25} , ...
            return_global (bool): in addition to the extrema, return the actual
                CBM (global minimum) and VBM (global maximum) w/ their k-point
            scissor (float): the amount by which the band gap is altered/
                scissored.

        Returns (dict): {'n': list of extrema fractional coordinates, 'p': same}
        """

        # TODO: Rewrite this to accept multiple bands

        bs = self._band_structure
        is_cb = iband > self._vbm_idx

        def to_cart(k):
            """Convert fractional k-points to cartesian coordinates in 1/nm."""
            return np.dot(bs.structure.lattice.reciprocal_lattice.matrix,
                          k) * 10

        def kpoints_are_separated(a_kpoint, list_kpoints):
            sym_kpoints = [k for lk in list_kpoints for dk in (lk, -lk)
                           for k in bs.get_sym_eq_kpoints(dk)]
            if sym_kpoints:
                return norm(to_cart(get_closest_k(
                    a_kpoint, sym_kpoints, return_diff=True))) > min_normdiff
            else:
                return True

        hsk = HighSymmKpath(bs.structure)
        kpoints = kpts_to_first_bz(hsk.get_kpoints(
            line_density=line_density)[0])
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

        # continuously filter to only include extrema separated in k-space
        tmp_extrema = []
        for kpoint_idx in extrema_idx:
            if kpoints_are_separated(kpoints[kpoint_idx], tmp_extrema):
                tmp_extrema.append(kpoints[kpoint_idx])

        # check to see if one of the actual high-symmetry k-points can be used
        # Is this to improve performance due to greater symmetry?
        # TODO: Check this is necessary
        all_high_sym_kpoints = [k for lk in hsk.kpath['kpoints'].values()
                                for dk in (lk, -lk)
                                for k in bs.get_sym_eq_kpoints(dk)]

        extrema = []
        for kpoint in tmp_extrema:
            high_sym_k = get_closest_k(kpoint, all_high_sym_kpoints)

            if norm(to_cart(high_sym_k - kpoint)) < min_normdiff / 10.0:
                extrema.append(high_sym_k)
            else:
                extrema.append(kpoint)
        extrema = np.array(extrema)

        # sort the extrema based on their energy (i.e. importance)
        band = self.get_energies(extrema, iband=iband, scissor=scissor)
        sorted_idx = np.argsort(band) if is_cb else np.argsort(band)[::-1]
        extrema = extrema[sorted_idx]

        if return_global:
            return extrema, global_extrema
        else:
            return extrema

    @staticmethod
    def _simplify_return_data(data: Sequence[Any]) -> Union[Any, Sequence[Any]]:
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
