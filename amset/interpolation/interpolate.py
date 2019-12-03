"""Band structure interpolation using BolzTraP2."""

import logging
import multiprocessing
import time

import numpy as np

from collections import defaultdict
from typing import Optional, Union, Tuple, List, Dict

from monty.json import MSONable
from scipy.ndimage import gaussian_filter1d

from BoltzTraP2 import units, sphere, fite

from amset.dos import FermiDos, get_dos
from amset.kpoints import (
    get_symmetry_equivalent_kpoints,
    get_kpoints,
    get_kpoint_mesh,
    similarity_transformation,
)
from amset.misc.log import log_time_taken, log_list
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.bandstructure import (
    BandStructure,
    BandStructureSymmLine,
)
from pymatgen.electronic_structure.dos import Dos
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.bandstructure import HighSymmKpath

from amset.data import AmsetData
from amset.constants import (
    hartree_to_ev,
    m_to_cm,
    A_to_m,
    hbar,
    e,
    m_e,
    bohr_to_angstrom,
    spin_name,
    numeric_types,
    int_types,
)

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__date__ = "June 21, 2019"

logger = logging.getLogger(__name__)


class Interpolater(MSONable):
    """Class to interpolate band structures based on BoltzTraP2.

    Details of the interpolation method are available in:

    3. Madsen, G. K. & Singh, D. J. Computer Physics Communications 175, 67–71
       (2006)
    3. Madsen, G. K., Carrete, J., and Verstraete, M. J. Computer Physics
        Communications 231, 140–145 (2018)

    Args:
        band_structure: A pymatgen band structure object.
        num_electrons: The number of electrons in the system.
        interpolation_factor: Factor used to determine the accuracy of the
            band structure interpolation. Also controls the k-point mesh density
            for :meth:`Interpolater.get_amset_data`.
        soc: Whether the system was calculated using spin–orbit coupling.
        magmom: The magnetic moments for each atom.
        mommat: The band structure derivatives.
        interpolate_projections: Whether to interpolate the band structure
            projections.
    """

    def __init__(
        self,
        band_structure: BandStructure,
        num_electrons: int,
        interpolation_factor: float = 20,
        soc: bool = False,
        magmom: Optional[np.ndarray] = None,
        mommat: Optional[np.ndarray] = None,
        interpolate_projections: bool = False,
    ):
        self._band_structure = band_structure
        self._num_electrons = num_electrons
        self._soc = soc
        self._spins = self._band_structure.bands.keys()
        self._interpolate_projections = interpolate_projections
        self.interpolation_factor = interpolation_factor
        self._lattice_matrix = band_structure.structure.lattice.matrix * units.Angstrom
        self._coefficients = {}
        self._projection_coefficients = defaultdict(dict)

        kpoints = np.array([k.frac_coords for k in band_structure.kpoints])
        atoms = AseAtomsAdaptor.get_atoms(band_structure.structure)

        logger.info("Getting band interpolation coefficients")

        t0 = time.perf_counter()
        self._equivalences = sphere.get_equivalences(
            atoms=atoms, nkpt=kpoints.shape[0] * interpolation_factor, magmom=magmom
        )

        # get the interpolation mesh used by BoltzTraP2
        self.interpolation_mesh = (
            2 * np.max(np.abs(np.vstack(self._equivalences)), axis=0) + 1
        )

        for spin in self._spins:
            energies = band_structure.bands[spin] * units.eV
            data = DFTData(kpoints, energies, self._lattice_matrix, mommat=mommat)
            self._coefficients[spin] = fite.fitde3D(data, self._equivalences)

        log_time_taken(t0)

        if self._interpolate_projections:
            logger.info("Getting projection interpolation coefficients")

            if not band_structure.projections:
                raise ValueError(
                    "interpolate_projections is True but band structure has no "
                    "projections"
                )

            for spin in self._spins:
                for label, projection in _get_projections(
                    band_structure.projections[spin]
                ):
                    data = DFTData(
                        kpoints, projection, self._lattice_matrix, mommat=mommat
                    )
                    self._projection_coefficients[spin][label] = fite.fitde3D(
                        data, self._equivalences
                    )
            log_time_taken(t0)

    def get_amset_data(
        self,
        energy_cutoff: Optional[float] = None,
        scissor: float = None,
        bandgap: float = None,
        symprec: float = 0.01,
        nworkers: int = -1,
    ) -> AmsetData:
        """Gets an AmsetData object using the interpolated bands.

        Note, the interpolation mesh is determined using by
        ``interpolate_factor`` option in the ``Inteprolater`` constructor.

        This method is much faster than the ``get_energies`` function but
        doesn't provide as much flexibility.

        The degree of parallelization is controlled by the ``nworkers`` option.

        Args:
            energy_cutoff: The energy cut-off to determine which bands are
                included in the interpolation. If the energy of a band falls
                within the cut-off at any k-point it will be included. For
                metals the range is defined as the Fermi level ± energy_cutoff.
                For gapped materials, the energy range is from the VBM -
                energy_cutoff to the CBM + energy_cutoff.
            scissor: The amount by which the band gap is scissored. Cannot
                be used in conjunction with the ``bandgap`` option. Has no
                effect for metallic systems.
            bandgap: Automatically adjust the band gap to this value. Cannot
                be used in conjunction with the ``scissor`` option. Has no
                effect for metallic systems.
            symprec: The symmetry tolerance used when determining the symmetry
                inequivalent k-points on which to interpolate.
            nworkers: The number of processors used to perform the
                interpolation. If set to ``-1``, the number of workers will
                be set to the number of CPU cores.

        Returns:
            The electronic structure (including energies, velocities, density of
            states and k-point information) as an AmsetData object.
        """
        is_metal = self._band_structure.is_metal()

        if is_metal and (bandgap or scissor):
            raise ValueError(
                "{} option set but system is metallic".format(
                    "bandgap" if bandgap else "scissor"
                )
            )

        if not self._interpolate_projections:
            raise ValueError(
                "Band structure projections needed to obtain full "
                "electronic structure. Reinitialise the "
                "interpolater with interpolate_projections=True"
            )

        nworkers = multiprocessing.cpu_count() if nworkers == -1 else nworkers

        str_kmesh = "x".join(map(str, self.interpolation_mesh))
        logger.info("Interpolation parameters:")
        log_list(
            [
                "k-point mesh: {}".format(str_kmesh),
                "energy cutoff: {} eV".format(energy_cutoff),
            ]
        )

        # only calculate the energies for the bands within the energy cutoff
        min_e, max_e = _get_energy_cutoffs(energy_cutoff, self._band_structure)

        energies = {}
        vvelocities = {}
        curvature = {}
        projections = defaultdict(dict)
        new_vb_idx = {}
        for spin in self._spins:
            bands = self._band_structure.bands[spin]
            ibands = np.any((bands > min_e) & (bands < max_e), axis=1)

            logger.info(
                "Interpolating {} bands {}-{}".format(
                    spin_name[spin],
                    np.where(ibands)[0].min() + 1,
                    np.where(ibands)[0].max() + 1,
                )
            )

            t0 = time.perf_counter()
            energies[spin], vvelocities[spin], curvature[spin] = fite.getBTPbands(
                self._equivalences,
                self._coefficients[spin][ibands],
                self._lattice_matrix,
                curvature=True,
                nworkers=nworkers,
            )
            log_time_taken(t0)

            if not is_metal:
                # Need the largest VB index. Sometimes the index of the band
                # containing the VBM is not the largest index of all VBs, so
                # find all bands with energies less than the VBM and count that
                # number
                vbm_energy = self._band_structure.get_vbm()["energy"]
                vb_idx = np.any(bands <= vbm_energy, axis=1).sum() - 1

                # need to know the index of the valence band after discounting
                # bands during the interpolation (i.e., if energy_cutoff is
                # set). As ibands is just a list of True/False, we can count the
                # number of Trues up to and including the VBM to get the new
                # number of valence bands
                new_vb_idx[spin] = sum(ibands[: vb_idx + 1]) - 1

            logger.info("Interpolating {} projections".format(spin_name[spin]))
            t0 = time.perf_counter()

            for label, proj_coeffs in self._projection_coefficients[spin].items():
                projections[spin][label] = fite.getBTPbands(
                    self._equivalences,
                    proj_coeffs[ibands],
                    self._lattice_matrix,
                    nworkers=nworkers,
                )[0]

            log_time_taken(t0)

        if is_metal:
            efermi = self._band_structure.efermi * units.eV
            scissor = 0.0
        else:
            energies, scissor = _shift_energies(
                energies,
                new_vb_idx,
                scissor=scissor,
                bandgap=bandgap,
                return_scissor=True,
            )

            # if material is semiconducting, set Fermi level to middle of gap
            efermi = _get_efermi(energies, new_vb_idx)

        # get the actual k-points used in the BoltzTraP2 interpolation
        # unfortunately, BoltzTraP2 doesn't expose this information so we
        # have to get it ourselves
        ir_kpts, _, full_kpts, ir_kpts_idx, ir_to_full_idx = get_kpoints(
            self.interpolation_mesh,
            self._band_structure.structure,
            symprec=symprec,
            return_full_kpoints=True,
        )

        return AmsetData(
            self._band_structure.structure,
            energies,
            vvelocities,
            projections,
            self.interpolation_mesh,
            full_kpts,
            ir_kpts,
            ir_kpts_idx,
            ir_to_full_idx,
            efermi,
            is_metal,
            self._soc,
            vb_idx=new_vb_idx,
            scissor=scissor,
            # curvature=curvature,
        )

    def get_amset_data_from_kpoints(
        self,
        kpoints: Union[np.ndarray, List[int], float, int],
        energy_cutoff: Optional[float] = None,
        scissor: float = None,
        bandgap: float = None,
        symprec: float = 0.01,
    ) -> AmsetData:
        """Gets an AmsetData object using the interpolated bands.

        Note, the interpolation mesh is determined using by
        ``interpolate_factor`` option in the ``Inteprolater`` constructor.

        This method is much faster than the ``get_energies`` function but
        doesn't provide as much flexibility.

        The degree of parallelization is controlled by the ``nworkers`` option.

        Args:
            kpoints: The k-points, either provided as a list of k-points (either with
                the shape (nkpoints, 3) or (nkpoints, 4) where the 4th column is the
                k-point weights). Alternatively, the k-points can be specified as a
                1x3 mesh, e.g.,``[6, 6, 6]`` from which the full Gamma centered mesh
                will be computed. Alternatively, if a single value is provided this will
                be treated as a real-space length cutoff and the k-point mesh dimensions
                generated automatically.
            energy_cutoff: The energy cut-off to determine which bands are
                included in the interpolation. If the energy of a band falls
                within the cut-off at any k-point it will be included. For
                metals the range is defined as the Fermi level ± energy_cutoff.
                For gapped materials, the energy range is from the VBM -
                energy_cutoff to the CBM + energy_cutoff.
            scissor: The amount by which the band gap is scissored. Cannot
                be used in conjunction with the ``bandgap`` option. Has no
                effect for metallic systems.
            bandgap: Automatically adjust the band gap to this value. Cannot
                be used in conjunction with the ``scissor`` option. Has no
                effect for metallic systems.
            symprec: The symmetry tolerance used when determining the symmetry
                inequivalent k-points on which to interpolate.

        Returns:
            The electronic structure (including energies, velocities, density of
            states and k-point information) as an AmsetData object.
        """
        is_metal = self._band_structure.is_metal()

        mesh_info = []
        if isinstance(kpoints, numeric_types) or isinstance(kpoints[0], int_types):
            # k-points is given as a cut-off or mesh
            if isinstance(kpoints, numeric_types):
                kpoints = get_kpoint_mesh(self._band_structure.structure, kpoints)

            interpolation_mesh = np.asarray(kpoints)
            str_kmesh = "x".join(map(str, kpoints))
            mesh_info.append("k-point mesh: {}".format(str_kmesh))

            _, _, kpoints, _, _ = get_kpoints(
                kpoints,
                self._band_structure.structure,
                symprec=symprec,
                return_full_kpoints=True,
                boltztrap_ordering=False,
            )
            weights = np.full(len(kpoints), 1 / len(kpoints))

        else:
            # the full list of k-points has been specified
            interpolation_mesh = None
            kpoints = np.asarray(kpoints)
            nkpoints = kpoints.shape[0]
            if kpoints.shape[-1] == 4:
                # kpoints have been provided with weights
                weights = kpoints[:, 3]
                kpoints = kpoints[:, :3]
            else:
                logger.warning(
                    "User supplied k-points have no weights... assuming uniform mesh"
                )
                weights = np.full(nkpoints, 1 / nkpoints)
            mesh_info.append(["# user supplied k-points: {}".format(nkpoints)])

        mesh_info.append("energy cutoff: {} eV".format(energy_cutoff))
        logger.info("Interpolation parameters:")
        log_list(mesh_info)

        energies, vvelocities, curvature, projections, mapping_info, efermi, vb_idx, scissor = self.get_energies(
            kpoints,
            energy_cutoff=energy_cutoff,
            scissor=scissor,
            bandgap=bandgap,
            return_velocity=True,
            return_curvature=True,
            return_projections=True,
            atomic_units=True,
            return_vel_outer_prod=True,
            return_kpoint_mapping=True,
            return_efermi=True,
            symprec=symprec,
            return_vb_idx=True,
            return_scissor=True,
        )

        ir_kpoints_idx = mapping_info["ir_kpoints_idx"]
        ir_to_full_idx = mapping_info["ir_to_full_idx"]
        ir_kpoints = kpoints[ir_kpoints_idx]

        return AmsetData(
            self._band_structure.structure,
            energies,
            vvelocities,
            projections,
            interpolation_mesh,
            kpoints,
            ir_kpoints,
            ir_kpoints_idx,
            ir_to_full_idx,
            efermi,
            is_metal,
            self._soc,
            vb_idx=vb_idx,
            scissor=scissor,
            kpoint_weights=weights,
            curvature=curvature,
        )

    def get_energies(
        self,
        kpoints: Union[np.ndarray, List],
        energy_cutoff: Optional[float] = None,
        scissor: float = None,
        bandgap: float = None,
        return_velocity: bool = False,
        return_curvature: bool = False,
        return_projections: bool = False,
        return_vel_outer_prod: bool = False,
        coords_are_cartesian: bool = False,
        atomic_units: bool = False,
        skip_coefficients: Optional[float] = None,
        symprec: Optional[float] = None,
        return_kpoint_mapping: bool = False,
        return_efermi: bool = False,
        return_vb_idx: bool = False,
        return_scissor: bool = False,
    ) -> Union[Dict[Spin, np.ndarray], Tuple[Dict[Spin, np.ndarray], ...]]:
        """Gets the interpolated energies for multiple k-points in a band.

        Note, the accuracy of the interpolation is dependant on the
        ``interpolate_factor`` used to initialize the Interpolater.

        Args:
            kpoints: The k-point coordinates.
            energy_cutoff: The energy cut-off to determine which bands are
                included in the interpolation. If the energy of a band falls
                within the cut-off at any k-point it will be included. For
                metals the range is defined as the Fermi level ± energy_cutoff.
                For gapped materials, the energy range is from the VBM -
                energy_cutoff to the CBM + energy_cutoff.
            scissor: The amount by which the band gap is scissored. Cannot
                be used in conjunction with the ``bandgap`` option. Has no
                effect for metallic systems.
            bandgap: Automatically adjust the band gap to this value. Cannot
                be used in conjunction with the ``scissor`` option. Has no
                effect for metallic systems.
            return_velocity: Whether to return the band velocities.
            return_curvature: Whether to return the band curvature (inverse effective
                mass).
            return_projections: Whether to return the interpolated projections.
            return_vel_outer_prod: Whether to return the outer product of
                velocity, as used by BoltzTraP2 to calculate transport
                properties.
            coords_are_cartesian: Whether the kpoints are in cartesian or
                fractional coordinates.
            atomic_units: Return the energies, velocities, and effective_massses
                in atomic units. If False, energies will be in eV, velocities in
                cm/s, and curvature in units of 1 / electron rest mass (1/m0).
            symprec: Symmetry precision. If set, symmetry will be used to
                reduce the nummber of calculated k-points and velocities.
            return_kpoint_mapping: If `True`, the kpoint symmetry mapping information
                will be returned. If ``symprec`` is None then all sites will be
                considered symmetry inequivalent.
            return_efermi: Whether to return the Fermi level with the unit
                determined by ``atomic_units``. If the system is semiconducting
                the Fermi level will be given in the middle of the band gap.
            return_vb_idx: Whether to return the index of the highest valence band
                in the interpolated bands. Will be returned as a dictionary of
                ``{spin: vb_idx}``.
            return_scissor: Whether to return the determined scissor value, given in
                Hartree.

        Returns:
            The band energies as dictionary of::

                {spin: energies}

            If ``return_velocity``, ``curvature`` or
            ``return_projections`` a tuple is returned, formatted as::

                (energies, Optional[velocities], Optional[curvature],
                 Optional[projections])

            The velocities and effective masses are given as the 1x3 trace and
            full 3x3 tensor, respectively (along cartesian directions). The
            projections are summed for each orbital type (s, p, d) across all
            atoms, and are given as::

                {spin: {orbital: projections}}
        """
        if self._band_structure.is_metal() and (bandgap or scissor):
            raise ValueError(
                "{} option set but system is metallic".format(
                    "bandgap" if bandgap else "scissor"
                )
            )

        if not self._interpolate_projections and return_projections:
            raise ValueError(
                "Band structure projections needed to obtain full "
                "electronic structure. Reinitialise the "
                "interpolater with interpolate_projections=True"
            )

        n_equivalences = len(self._equivalences)
        if not skip_coefficients or skip_coefficients > 1:
            skip = n_equivalences
        else:
            skip = int(skip_coefficients * n_equivalences)

        # only calculate the energies for the bands within the energy cutoff
        min_e, max_e = _get_energy_cutoffs(energy_cutoff, self._band_structure)
        lattice = self._band_structure.structure.lattice

        if coords_are_cartesian:
            kpoints = lattice.reciprocal_lattice.get_fractional_coords(kpoints)

        nkpoints = len(kpoints)

        if symprec:
            logger.info("Reducing # k-points using symmetry")
            kpoints, weights, ir_kpoints_idx, ir_to_full_idx, _, rot_mapping = get_symmetry_equivalent_kpoints(
                self._band_structure.structure,
                kpoints,
                symprec=symprec,
                return_inverse=True,
            )
            similarity_matrix = np.array(
                [
                    similarity_transformation(lattice.reciprocal_lattice.matrix, r)
                    for r in rot_mapping
                ]
            )
            # similarity_matrix = rot_mapping

            inv_similarity_matrix = np.array(
                [np.linalg.inv(s) for s in similarity_matrix]
            )

            log_list(
                [
                    "# original k-points: {}".format(nkpoints),
                    "# reduced k-points {}".format(len(kpoints)),
                ]
            )
        else:
            kpoints = np.asarray(kpoints)
            nkpoints = kpoints.shape[0]
            ir_kpoints_idx = np.arange(nkpoints)
            ir_to_full_idx = np.arange(nkpoints)
            weights = np.full(nkpoints, 1 / nkpoints)

        energies = {}
        velocities = {}
        curvature = {}
        projections = defaultdict(dict)
        new_vb_idx = {}
        for spin in self._spins:
            bands = self._band_structure.bands[spin]
            ibands = np.any((bands > min_e) & (bands < max_e), axis=1)

            logger.info(
                "Interpolating {} bands {}-{}".format(
                    spin_name[spin],
                    np.where(ibands)[0].min() + 1,
                    np.where(ibands)[0].max() + 1,
                )
            )

            t0 = time.perf_counter()
            fitted = fite.getBands(
                kpoints,
                self._equivalences[:skip],
                self._lattice_matrix,
                self._coefficients[spin][ibands, :skip],
                curvature=return_curvature,
            )
            log_time_taken(t0)

            energies[spin] = fitted[0]
            velocities[spin] = fitted[1]

            if symprec:
                energies[spin] = energies[spin][:, ir_to_full_idx]

                # apply rotation matrices to the velocities at the symmetry
                # reduced k-points, to get the velocities for the full
                # original mesh (this is just the dot product of the velocity
                # and appropriate rotation matrix. The weird ordering of the
                # indices is because the velocities has the shape
                # (3, nbands, nkpoints)
                # velocities[spin] = np.einsum(
                #     "kij,jkl->lij",
                #     velocities[spin][:, :, ir_to_full_idx],
                #     similarity_matrix,
                # )
                velocities[spin] = np.einsum(
                    "jkl,kij->lij",
                    similarity_matrix,
                    velocities[spin][:, :, ir_to_full_idx],
                )

            if not self._band_structure.is_metal():
                # valence bands are all bands that contain energies less than efermi
                vbs = (bands < self._band_structure.efermi).any(axis=1)
                vb_idx = np.where(vbs)[0].max()

                # need to know the index of the valence band after discounting
                # bands during the interpolation. As ibands is just a list of
                # True/False, we can count the number of Trues included up to
                # and including the VBM to get the new number of valence bands
                new_vb_idx[spin] = sum(ibands[: vb_idx + 1]) - 1

            if return_vel_outer_prod:
                # calculate the outer produce of velocities with itself
                # this code is adapted from BoltzTraP2.fite
                iu0 = np.triu_indices(3)
                il1 = np.tril_indices(3, -1)
                iu1 = np.triu_indices(3, 1)

                velocities[spin] = velocities[spin].transpose((1, 0, 2))
                vvband = np.zeros((len(velocities[spin]), 3, 3, nkpoints))
                vvband[:, iu0[0], iu0[1]] = (
                    velocities[spin][:, iu0[0]] * velocities[spin][:, iu0[1]]
                )
                vvband[:, il1[0], il1[1]] = vvband[:, iu1[0], iu1[1]]
                velocities[spin] = vvband

            if return_curvature:
                curvature[spin] = fitted[2]

                # make curvature have the shape ((nbands, nkpoints, 3, 3)
                curvature[spin] = curvature[spin].transpose((2, 3, 0, 1))

                if symprec:
                    curvature[spin] = curvature[spin][:, ir_to_full_idx, ...]
                    new_curvature = np.empty(curvature[spin].shape)
                    for b_idx, k_idx in np.ndindex(curvature[spin].shape[:2]):
                        new_curvature[b_idx, k_idx] = np.dot(
                            inv_similarity_matrix[k_idx],
                            np.dot(
                                curvature[spin][b_idx, k_idx], similarity_matrix[k_idx]
                            ),
                        )
                    curvature[spin] = new_curvature

            if not atomic_units:
                energies[spin] = energies[spin] / units.eV
                velocities[spin] = _convert_velocities(velocities[spin], lattice.matrix)

                if return_curvature:
                    curvature[spin] = _convert_curvature(curvature[spin])

            if return_projections:
                logger.info("Interpolating {} projections".format(spin_name[spin]))

                t0 = time.perf_counter()
                for label, proj_coeffs in self._projection_coefficients[spin].items():
                    projections[spin][label] = fite.getBands(
                        kpoints,
                        self._equivalences[:skip],
                        self._lattice_matrix,
                        proj_coeffs[ibands, :skip],
                        curvature=False,
                    )[0]

                    if symprec:
                        projections[spin][label] = projections[spin][label][
                            :, ir_to_full_idx
                        ]

                log_time_taken(t0)

        if not self._band_structure.is_metal():
            energies, scissor = _shift_energies(
                energies,
                new_vb_idx,
                scissor=scissor,
                bandgap=bandgap,
                return_scissor=True,
            )
        else:
            scissor = 0

        if not (
            return_velocity
            or return_curvature
            or return_projections
            or return_kpoint_mapping
            or return_efermi
            or return_vb_idx
            or return_scissor
        ):
            return energies

        to_return = [energies]

        if return_velocity:
            to_return.append(velocities)

        if return_curvature:
            to_return.append(curvature)

        if return_projections:
            to_return.append(projections)

        if symprec and return_kpoint_mapping:
            to_return.append(
                {
                    "weights": weights,
                    "ir_kpoints_idx": ir_kpoints_idx,
                    "ir_to_full_idx": ir_to_full_idx,
                }
            )

        if return_efermi:
            if self._band_structure.is_metal():
                efermi = self._band_structure.efermi
                if atomic_units:
                    efermi *= units.eV
            else:
                # if semiconducting, set Fermi level to middle of gap
                efermi = _get_efermi(energies, new_vb_idx)

            to_return.append(efermi)

        if return_vb_idx:
            to_return.append(new_vb_idx)

        if return_scissor:
            to_return.append(scissor)

        return tuple(to_return)

    def get_dos(
        self,
        kpoint_mesh: Union[float, int, List[int]],
        energy_cutoff: Optional[float] = None,
        scissor: Optional[float] = None,
        bandgap: Optional[float] = None,
        estep: float = 0.01,
        width: float = 0.05,
        symprec: float = 0.01,
        fermi_dos: bool = False,
    ) -> Union[Dos, FermiDos]:
        """Calculates the density of states using the interpolated bands.

        Args:
            kpoint_mesh: The k-point mesh as a 1x3 array. E.g.,``[6, 6, 6]``.
                Alternatively, if a single value is provided this will be
                treated as a reciprocal density and the k-point mesh dimensions
                generated automatically.
            energy_cutoff: The energy cut-off to determine which bands are
                included in the interpolation. If the energy of a band falls
                within the cut-off at any k-point it will be included. For
                metals the range is defined as the Fermi level ± energy_cutoff.
                For gapped materials, the energy range is from the VBM -
                energy_cutoff to the CBM + energy_cutoff.
            scissor: The amount by which the band gap is scissored. Cannot
                be used in conjunction with the ``bandgap`` option. Has no
                effect for metallic systems.
            bandgap: Automatically adjust the band gap to this value. Cannot
                be used in conjunction with the ``scissor`` option. Has no
                effect for metallic systems.
            estep: The energy step, where smaller numbers give more
                accuracy but are more expensive.
            width: The gaussian smearing width in eV.
            symprec: The symmetry tolerance used when determining the symmetry
                inequivalent k-points on which to interpolate.
            fermi_dos: Whether to return a FermiDos object, instead of a regular
                Dos.

        Returns:
            The density of states.
        """
        kpoints, weights = get_kpoints(
            kpoint_mesh, self._band_structure.structure, symprec=symprec
        )

        energies, efermi = self.get_energies(
            kpoints,
            scissor=scissor,
            bandgap=bandgap,
            energy_cutoff=energy_cutoff,
            atomic_units=False,
            return_efermi=True,
        )

        emin = np.min([np.min(spin_eners) for spin_eners in energies.values()])
        emin -= width * 5 if width else 0.1
        emax = np.max([np.max(spin_eners) for spin_eners in energies.values()])
        emax += width * 5 if width else 0.1
        epoints = int(round((emax - emin) / estep))

        dos = {}
        emesh = None
        for spin in self._spins:
            kpoint_weights = np.tile(
                weights / np.sum(weights), (len(energies[spin]), 1)
            )
            emesh, dos[spin] = get_dos(
                energies[spin].T,
                erange=(emin, emax),
                npts=epoints,
                weights=kpoint_weights.T,
            )

            if width:
                dos[spin] = gaussian_filter1d(dos[spin], width / (emesh[1] - emesh[0]))

        if fermi_dos:
            return FermiDos(
                efermi, emesh, dos, self._band_structure.structure, atomic_units=False
            )
        else:
            return Dos(efermi, emesh, dos)

    def get_band_structure(
        self,
        kpoint_mesh: Union[float, int, List[int]],
        energy_cutoff: Optional[float] = None,
        scissor: Optional[float] = None,
        bandgap: Optional[float] = None,
        symprec: float = 0.01,
    ) -> BandStructure:
        """Calculates the density of states using the interpolated bands.

        Args:
            kpoint_mesh: The k-point mesh as a 1x3 array. E.g.,``[6, 6, 6]``.
                Alternatively, if a single value is provided this will be
                treated as a reciprocal density and the k-point mesh dimensions
                generated automatically.
            energy_cutoff: The energy cut-off to determine which bands are
                included in the interpolation. If the energy of a band falls
                within the cut-off at any k-point it will be included. For
                metals the range is defined as the Fermi level ± energy_cutoff.
                For gapped materials, the energy range is from the VBM -
                energy_cutoff to the CBM + energy_cutoff.
            scissor: The amount by which the band gap is scissored. Cannot
                be used in conjunction with the ``bandgap`` option. Has no
                effect for metallic systems.
            bandgap: Automatically adjust the band gap to this value. Cannot
                be used in conjunction with the ``scissor`` option. Has no
                effect for metallic systems.
            symprec: The symmetry tolerance used when determining the symmetry
                inequivalent k-points on which to interpolate.

        Returns:
            The density of states.
        """
        ir_kpoints, weights, full_kpoints, ir_kpoints_idx, ir_to_full_idx = get_kpoints(
            kpoint_mesh,
            self._band_structure.structure,
            symprec=symprec,
            return_full_kpoints=True,
        )

        energies = self.get_energies(
            ir_kpoints,
            scissor=scissor,
            bandgap=bandgap,
            energy_cutoff=energy_cutoff,
            atomic_units=True,
        )

        energies = {
            s: bands[:, ir_to_full_idx] / units.eV for s, bands in energies.items()
        }

        return BandStructure(
            full_kpoints,
            energies,
            self._band_structure.structure.lattice,
            self._band_structure.efermi,
            coords_are_cartesian=True,
            structure=self._band_structure.structure,
        )

    def get_line_mode_band_structure(
        self,
        line_density: int = 50,
        energy_cutoff: Optional[float] = None,
        scissor: Optional[float] = None,
        bandgap: Optional[float] = None,
        symprec: float = 0.01,
    ) -> BandStructureSymmLine:
        """Gets the interpolated band structure along high symmetry directions.

        Args:
            line_density: The maximum number of k-points between each two
                consecutive high-symmetry k-points
            energy_cutoff: The energy cut-off to determine which bands are
                included in the interpolation. If the energy of a band falls
                within the cut-off at any k-point it will be included. For
                metals the range is defined as the Fermi level ± energy_cutoff.
                For gapped materials, the energy range is from the VBM -
                energy_cutoff to the CBM + energy_cutoff.
            scissor: The amount by which the band gap is scissored. Cannot
                be used in conjunction with the ``bandgap`` option. Has no
                effect for metallic systems.
            bandgap: Automatically adjust the band gap to this value. Cannot
                be used in conjunction with the ``scissor`` option. Has no
                effect for metallic systems.
            symprec: The symmetry tolerance used to determine the space group
                and high-symmetry path.

        Returns:
            The line mode band structure.
        """

        hsk = HighSymmKpath(self._band_structure.structure, symprec=symprec)
        kpoints, labels = hsk.get_kpoints(
            line_density=line_density, coords_are_cartesian=True
        )
        labels_dict = {
            label: kpoint for kpoint, label in zip(kpoints, labels) if label != ""
        }

        energies = self.get_energies(
            kpoints,
            scissor=scissor,
            bandgap=bandgap,
            atomic_units=False,
            energy_cutoff=energy_cutoff,
            coords_are_cartesian=True,
        )

        return BandStructureSymmLine(
            kpoints,
            energies,
            self._band_structure.structure.lattice,
            self._band_structure.efermi,
            labels_dict,
            coords_are_cartesian=True,
        )


class DFTData(object):
    """DFTData object used for BoltzTraP2 interpolation.

    Note that the units used by BoltzTraP are different to those used by VASP.

    Args:
        kpoints: The k-points in fractional coordinates.
        energies: The band energies in Hartree, formatted as (nbands, nkpoints).
        lattice_matrix: The lattice matrix in Bohr^3.
        mommat: The band structure derivatives.
    """

    def __init__(
        self,
        kpoints: np.ndarray,
        energies: np.ndarray,
        lattice_matrix: np.ndarray,
        mommat: Optional[np.ndarray] = None,
    ):
        self.kpoints = kpoints
        self.ebands = energies
        self.lattice_matrix = lattice_matrix
        self.volume = np.abs(np.linalg.det(self.lattice_matrix))
        self.mommat = mommat

    def get_lattvec(self) -> np.ndarray:
        """Get the lattice matrix. This method is required by BoltzTraP2."""
        return self.lattice_matrix


def _shift_energies(
    energies: Dict[Spin, np.ndarray],
    vb_idx: Dict[Spin, int],
    scissor: Optional[float] = None,
    bandgap: Optional[float] = None,
    return_scissor: bool = False,
) -> Union[Dict[Spin, np.ndarray], Tuple[Dict[Spin, np.ndarray], float]]:
    """Shift the band energies based on the scissor or bandgap parameter.

    Args:
        energies: The band energies in Hartree, given for each Spin channel.
        vb_idx: The band index of the valence band maximum in the energies
            array, given for each Spin channel.
        scissor: The amount by which the band gap is scissored. Cannot
            be used in conjunction with the ``bandgap`` option. Has no
            effect for metallic systems.
        bandgap: Automatically adjust the band gap to this value. Cannot
            be used in conjunction with the ``scissor`` option. Has no
            effect for metallic systems.
        return_scissor: Whether to return the determined scissor value, given in
            Hartree.

    Returns:
        The energies, shifted according to ``scissor`` or ``bandgap``. If
        return_scissor is True, a tuple of (energies, scissor) is returned.
    """

    if scissor and bandgap:
        raise ValueError("scissor and bandgap cannot be set simultaneously")

    cb_idx = {s: v + 1 for s, v in vb_idx.items()}

    if bandgap:
        interp_bandgap = (
            min([energies[s][cb_idx[s] :].min() for s in energies])
            - max([energies[s][: cb_idx[s]].max() for s in energies])
        ) / units.eV

        scissor = bandgap - interp_bandgap
        logger.debug(
            "Bandgap set to {:.3f} eV, automatically scissoring by "
            "{:.3f} eV".format(bandgap, scissor)
        )

    if scissor:
        scissor *= units.eV  # convert to Hartree
        for spin in energies:
            energies[spin][: cb_idx[spin]] -= scissor / 2
            energies[spin][cb_idx[spin] :] += scissor / 2

    if return_scissor:
        return energies, scissor
    else:
        return energies


def _convert_velocities(
    velocities: np.ndarray, lattice_matrix: np.ndarray
) -> np.ndarray:
    """Convert velocities from atomic units to cm/s.

    Args:
        velocities: The velocities in atomic units.
        lattice_matrix: The lattice matrix in Angstrom.

    Returns:
        The velocities in cm/s.
    """
    matrix_norm = lattice_matrix / np.linalg.norm(lattice_matrix)

    factor = hartree_to_ev * m_to_cm * A_to_m / (hbar * bohr_to_angstrom)
    velocities = velocities.transpose((1, 0, 2))

    velocities = np.abs(np.matmul(matrix_norm, velocities)) * factor
    velocities = velocities.transpose((0, 2, 1))

    return velocities


def _convert_curvature(curvature: np.ndarray) -> np.ndarray:
    """Convert curvature from atomic units to units of electron rest mass.

    Args:
        curvature: The band curvature in atomic units.

    Returns:
        The curvature (i.e., inverse effective mass) in units of 1 / electron rest mass.
    """
    factor = bohr_to_angstrom ** 2 * e * hbar ** 2 / (hartree_to_ev * A_to_m ** 2 * m_e)
    return curvature / factor


def _get_projections(projections: np.ndarray) -> Tuple[Tuple[str, np.ndarray], ...]:
    """Extracts and sums the band structure projections for a band.

    Args:
        projections: The projections for a band.

    Returns:
        The projection labels and orbital projections, as::

            ("s", s_orbital_projections), ("p", p_orbital_projections)
    """
    s_orbital = np.sum(projections, axis=3)[:, :, 0]

    if projections.shape[2] > 5:
        # lm decomposed projections therefore sum across px, py, and pz
        p_orbital = np.sum(np.sum(projections, axis=3)[:, :, 1:4], axis=2)
    else:
        p_orbital = np.sum(projections, axis=3)[:, :, 1]

    return ("s", s_orbital), ("p", p_orbital)


def _get_energy_cutoffs(
    energy_cutoff: float, band_structure: BandStructure
) -> Tuple[float, float]:
    if energy_cutoff and band_structure.is_metal():
        min_e = band_structure.efermi - energy_cutoff
        max_e = band_structure.efermi + energy_cutoff
    elif energy_cutoff:
        min_e = band_structure.get_vbm()["energy"] - energy_cutoff
        max_e = band_structure.get_cbm()["energy"] + energy_cutoff
    else:
        min_e = min(
            [band_structure.bands[spin].min() for spin in band_structure.bands.keys()]
        )
        max_e = max(
            [band_structure.bands[spin].max() for spin in band_structure.bands.keys()]
        )

    return min_e, max_e


def _get_efermi(energies: Dict[Spin, np.ndarray], vb_idx: Dict[Spin, int]) -> float:
    e_vbm = max(
        [np.max(energies[spin][: vb_idx[spin] + 1]) for spin in energies.keys()]
    )
    e_cbm = min(
        [np.min(energies[spin][vb_idx[spin] + 1 :]) for spin in energies.keys()]
    )
    return (e_vbm + e_cbm) / 2
