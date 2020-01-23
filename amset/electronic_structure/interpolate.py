"""Band structure electronic_structure using BolzTraP2."""

import logging
import multiprocessing as mp
import multiprocessing.sharedctypes
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from BoltzTraP2 import fite, sphere
from BoltzTraP2.fite import FFTc, FFTev, Second
from monty.json import MSONable

from amset.constants import amset_defaults as defaults, hartree_to_ev
from amset.constants import angstrom_to_bohr, bohr_to_cm, ev_to_hartree, spin_name
from amset.core.data import AmsetData
from amset.electronic_structure.dos import FermiDos
from amset.electronic_structure.overlap import OverlapCalculator
from amset.electronic_structure.kpoints import (
    get_kpoints_tetrahedral,
    get_symmetry_equivalent_kpoints,
    similarity_transformation,
    sort_boltztrap_to_spglib,
)
from amset.log import log_list, log_time_taken
from amset.electronic_structure.tetrahedron import TetrahedralBandStructure
from pymatgen import Structure
from pymatgen.electronic_structure.bandstructure import (
    BandStructure,
    BandStructureSymmLine,
)
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.dos import Dos
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.bandstructure import HighSymmKpath

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"
__date__ = "June 21, 2019"

logger = logging.getLogger(__name__)


class Interpolater(MSONable):
    """Class to interpolate band structures based on BoltzTraP2.

    Details of the electronic_structure method are available in:

    3. Madsen, G. K. & Singh, D. J. Computer Physics Communications 175, 67–71
       (2006)
    3. Madsen, G. K., Carrete, J., and Verstraete, M. J. Computer Physics
        Communications 231, 140–145 (2018)

    Args:
        band_structure: A pymatgen band structure object.
        num_electrons: The number of electrons in the system.
        interpolation_factor: Factor used to determine the accuracy of the
            band structure electronic_structure. Also controls the k-point mesh density
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
        interpolation_factor: float = defaults["interpolation_factor"],
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
        self._lattice_matrix = (
            band_structure.structure.lattice.matrix.T * angstrom_to_bohr
        )
        self._coefficients = {}
        self._projection_coefficients = defaultdict(dict)

        kpoints = np.array([k.frac_coords for k in band_structure.kpoints])
        atoms = AseAtomsAdaptor.get_atoms(band_structure.structure)

        logger.info("Getting band electronic_structure coefficients")

        t0 = time.perf_counter()
        self._equivalences = sphere.get_equivalences(
            atoms=atoms, nkpt=kpoints.shape[0] * interpolation_factor, magmom=magmom
        )

        # get the electronic_structure mesh used by BoltzTraP2
        self.interpolation_mesh = (
            2 * np.max(np.abs(np.vstack(self._equivalences)), axis=0) + 1
        )

        for spin in self._spins:
            energies = band_structure.bands[spin] * ev_to_hartree
            data = DFTData(kpoints, energies, self._lattice_matrix, mommat=mommat)
            self._coefficients[spin] = fite.fitde3D(data, self._equivalences)

        log_time_taken(t0)

        if self._interpolate_projections:
            logger.info("Getting projection electronic_structure coefficients")

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
        symprec: float = defaults["symprec"],
        nworkers: int = defaults["nworkers"],
    ) -> AmsetData:
        """Gets an AmsetData object using the interpolated bands.

        Note, the electronic_structure mesh is determined using by
        ``interpolate_factor`` option in the ``Inteprolater`` constructor.

        This method is much faster than the ``get_energies`` function but
        doesn't provide as much flexibility.

        The degree of parallelization is controlled by the ``nworkers`` option.

        Args:
            energy_cutoff: The energy cut-off to determine which bands are
                included in the electronic_structure. If the energy of a band falls
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
                electronic_structure. If set to ``-1``, the number of workers will
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
        effective_mass = {}
        projections = defaultdict(dict)
        new_vb_idx = {}
        overlap_projections = {}
        forgotten_electrons = 0
        for spin in self._spins:
            bands = self._band_structure.bands[spin]
            ibands = np.any((bands > min_e) & (bands < max_e), axis=1)

            # these are bands beneath the Fermi level that are dropped
            forgotten_electrons += np.where(ibands)[0].min()

            logger.info(
                "Interpolating {} bands {}-{}".format(
                    spin_name[spin],
                    np.where(ibands)[0].min() + 1,
                    np.where(ibands)[0].max() + 1,
                )
            )

            t0 = time.perf_counter()
            energies[spin], vvelocities[spin], effective_mass[spin] = get_bands_fft(
                self._equivalences,
                self._coefficients[spin][ibands],
                self._lattice_matrix,
                return_effective_mass=True,
                nworkers=nworkers,
            )
            log_time_taken(t0)

            if not is_metal:
                # valence bands are all bands that contain energies less than efermi
                vbs = (bands < self._band_structure.efermi).any(axis=1)
                vb_idx = np.where(vbs)[0].max()

                # need to know the index of the valence band after discounting
                # bands during the electronic_structure. As ibands is just a list of
                # True/False, we can count the number of Trues included up to
                # and including the VBM to get the new number of valence bands
                new_vb_idx[spin] = sum(ibands[: vb_idx + 1]) - 1

            logger.info("Interpolating {} projections".format(spin_name[spin]))
            t0 = time.perf_counter()

            for label, proj_coeffs in self._projection_coefficients[spin].items():
                projections[spin][label] = get_bands_fft(
                    self._equivalences,
                    proj_coeffs[ibands],
                    self._lattice_matrix,
                    nworkers=nworkers,
                )[0]

            log_time_taken(t0)
            overlap_projections[spin] = self._band_structure.projections[spin][ibands]

        if not self._soc and len(self._spins) == 1:
            forgotten_electrons *= 2
        nelectrons = self._num_electrons - forgotten_electrons

        if is_metal:
            efermi = self._band_structure.efermi * ev_to_hartree
        else:
            energies = _shift_energies(
                energies, new_vb_idx, scissor=scissor, bandgap=bandgap
            )

            # if material is semiconducting, set Fermi level to middle of gap
            efermi = _get_efermi(energies, new_vb_idx)

        logger.info("Generating tetrahedron mesh")
        # get the actual k-points used in the BoltzTraP2 electronic_structure
        # unfortunately, BoltzTraP2 doesn't expose this information so we
        # have to get it ourselves
        ir_kpts, _, full_kpts, ir_kpts_idx, ir_to_full_idx, tetrahedra, *ir_tetrahedra_info = get_kpoints_tetrahedral(
            self.interpolation_mesh, self._band_structure.structure, symprec=symprec
        )

        # BoltzTraP2 and spglib give k-points in different orders. We need to use the
        # spglib ordering to make the tetrahedron method work, so get the indices
        # that will sort from BoltzTraP2 order to spglib order
        sort_idx = sort_boltztrap_to_spglib(full_kpts)

        energies = {s: ener[:, sort_idx] for s, ener in energies.items()}
        vvelocities = {s: vv[:, :, :, sort_idx] for s, vv in vvelocities.items()}
        projections = {
            s: {label: p[:, sort_idx] for label, p in proj.items()}
            for s, proj in projections.items()
        }

        original_structure = self._band_structure.structure
        atomic_structure = Structure(
            original_structure.lattice.matrix * angstrom_to_bohr,
            original_structure.species,
            original_structure.frac_coords,
        )

        band_centers = get_band_centers(full_kpts, energies, new_vb_idx, efermi)
        orig_kpoints = np.array([k.frac_coords for k in self._band_structure.kpoints])
        overlap_calculator = OverlapCalculator(
            atomic_structure, orig_kpoints, overlap_projections, band_centers
        )

        return AmsetData(
            atomic_structure,
            energies,
            vvelocities,
            effective_mass,
            projections,
            self.interpolation_mesh,
            full_kpts,
            ir_kpts,
            ir_kpts_idx,
            ir_to_full_idx,
            tetrahedra,
            ir_tetrahedra_info,
            efermi,
            nelectrons,
            is_metal,
            self._soc,
            overlap_calculator,
            vb_idx=new_vb_idx,
        )

    def get_energies(
        self,
        kpoints: Union[np.ndarray, List],
        energy_cutoff: Optional[float] = None,
        scissor: float = None,
        bandgap: float = None,
        return_velocity: bool = False,
        return_curvature: bool = False,
        coords_are_cartesian: bool = False,
        atomic_units: bool = False,
        symprec: Optional[float] = defaults["symprec"],
        return_efermi: bool = False,
        return_vb_idx: bool = False,
    ) -> Union[Dict[Spin, np.ndarray], Tuple[Dict[Spin, np.ndarray], ...]]:
        """Gets the interpolated energies for multiple k-points in a band.

        Note, the accuracy of the electronic_structure is dependant on the
        ``interpolate_factor`` used to initialize the Interpolater.

        Args:
            kpoints: The k-point coordinates.
            energy_cutoff: The energy cut-off to determine which bands are
                included in the electronic_structure. If the energy of a band falls
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
            coords_are_cartesian: Whether the kpoints are in cartesian or
                fractional coordinates.
            atomic_units: Return the energies, velocities, and effective_massses
                in atomic units. If False, energies will be in eV, velocities in
                cm/s, and curvature in units of 1 / electron rest mass (1/m0).
            symprec: Symmetry precision. If set, symmetry will be used to
                reduce the nummber of calculated k-points and velocities.
            return_efermi: Whether to return the Fermi level with the unit
                determined by ``atomic_units``. If the system is semiconducting
                the Fermi level will be given in the middle of the band gap.
            return_vb_idx: Whether to return the index of the highest valence band
                in the interpolated bands. Will be returned as a dictionary of
                ``{spin: vb_idx}``.

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
            ir_to_full_idx = np.arange(nkpoints)

        energies = {}
        velocities = {}
        curvature = {}
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
                self._equivalences,
                self._lattice_matrix,
                self._coefficients[spin][ibands],
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
                # bands during the electronic_structure. As ibands is just a list of
                # True/False, we can count the number of Trues included up to
                # and including the VBM to get the new number of valence bands
                new_vb_idx[spin] = sum(ibands[: vb_idx + 1]) - 1

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
                energies[spin] = energies[spin] * hartree_to_ev
                velocities[spin] = _convert_velocities(velocities[spin], lattice.matrix)

        if not self._band_structure.is_metal():
            energies = _shift_energies(
                energies, new_vb_idx, scissor=scissor, bandgap=bandgap
            )

        if not (return_velocity or return_curvature or return_efermi or return_vb_idx):
            return energies

        to_return = [energies]

        if return_velocity:
            to_return.append(velocities)

        if return_curvature:
            to_return.append(curvature)

        if return_efermi:
            if self._band_structure.is_metal():
                efermi = self._band_structure.efermi
                if atomic_units:
                    efermi *= ev_to_hartree
            else:
                # if semiconducting, set Fermi level to middle of gap
                efermi = _get_efermi(energies, new_vb_idx)

            to_return.append(efermi)

        if return_vb_idx:
            to_return.append(new_vb_idx)

        return tuple(to_return)

    def get_dos(
        self,
        kpoint_mesh: Union[float, int, List[int]],
        energy_cutoff: Optional[float] = None,
        scissor: Optional[float] = None,
        bandgap: Optional[float] = None,
        estep: float = defaults["dos_estep"],
        symprec: float = defaults["symprec"],
    ) -> Union[Dos, FermiDos]:
        """Calculates the density of states using the interpolated bands.

        Args:
            kpoint_mesh: The k-point mesh as a 1x3 array. E.g.,``[6, 6, 6]``.
                Alternatively, if a single value is provided this will be
                treated as a reciprocal density and the k-point mesh dimensions
                generated automatically.
            energy_cutoff: The energy cut-off to determine which bands are
                included in the electronic_structure. If the energy of a band falls
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
            symprec: The symmetry tolerance used when determining the symmetry
                inequivalent k-points on which to interpolate.

        Returns:
            The density of states.
        """
        ir_kpts, _, full_kpts, ir_kpts_idx, ir_to_full_idx, tetrahedra, *ir_tetrahedra_info = get_kpoints_tetrahedral(
            kpoint_mesh, self._band_structure.structure, symprec=symprec
        )

        energies, efermi = self.get_energies(
            ir_kpts,
            scissor=scissor,
            bandgap=bandgap,
            energy_cutoff=energy_cutoff,
            atomic_units=False,
            return_efermi=True,
        )

        tetrahedral_band_structure = TetrahedralBandStructure(
            {s: e[:, ir_to_full_idx] for s, e in energies.items()},
            full_kpts,
            tetrahedra,
            self._band_structure.structure,
            ir_kpts_idx,
            ir_to_full_idx,
            *ir_tetrahedra_info
        )

        emin = np.min([np.min(spin_eners) for spin_eners in energies.values()])
        emax = np.max([np.max(spin_eners) for spin_eners in energies.values()])
        epoints = int(round((emax - emin) / estep))
        energies = np.linspace(emin, emax, epoints)

        _, dos = tetrahedral_band_structure.get_density_of_states(energies)

        return FermiDos(
            efermi, energies, dos, self._band_structure.structure, atomic_units=False
        )

    def get_line_mode_band_structure(
        self,
        line_density: int = 50,
        energy_cutoff: Optional[float] = None,
        scissor: Optional[float] = None,
        bandgap: Optional[float] = None,
        symprec: float = defaults["symprec"],
    ) -> BandStructureSymmLine:
        """Gets the interpolated band structure along high symmetry directions.

        Args:
            line_density: The maximum number of k-points between each two
                consecutive high-symmetry k-points
            energy_cutoff: The energy cut-off to determine which bands are
                included in the electronic_structure. If the energy of a band falls
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
    """DFTData object used for BoltzTraP2 electronic_structure.

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
        ) * hartree_to_ev

        scissor = bandgap - interp_bandgap
        logger.debug(
            "Bandgap set to {:.3f} eV, automatically scissoring by "
            "{:.3f} eV".format(bandgap, scissor)
        )

    if scissor:
        scissor *= ev_to_hartree
        for spin in energies:
            energies[spin][: cb_idx[spin]] -= scissor / 2
            energies[spin][cb_idx[spin] :] += scissor / 2

    return energies


def _convert_velocities(
    velocities: np.ndarray, lattice_matrix: np.ndarray
) -> np.ndarray:
    """Convert velocities from atomic units to cm/s.

    # TODO: Check this...

    Args:
        velocities: The velocities in atomic units.
        lattice_matrix: The lattice matrix in Angstrom.

    Returns:
        The velocities in cm/s.
    """
    velocities *= bohr_to_cm * Second
    return velocities


def _get_projections(projections: np.ndarray) -> Tuple[Tuple[str, np.ndarray], ...]:
    """Extracts and sums the band structure projections for a band.

    Args:
        projections: The projections for a band.

    Returns:
        The projection labels and orbital projections, as::

            ("s", s_orbital_projections), ("p", d_orbital_projections)
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


def get_bands_fft(
    equivalences,
    coeffs,
    lattvec,
    return_effective_mass=False,
    nworkers=defaults["nworkers"],
):
    """Rebuild the full energy bands from the electronic_structure coefficients.

    Args:
        equivalences: list of k-point equivalence classes in direct coordinates
        coeffs: electronic_structure coefficients
        lattvec: lattice vectors of the system
        return_effective_mass: Whether to calculate the effective mass.
        nworkers: number of working processes to span

    Returns:
        A 3-tuple (eband, vvband, cband): energy bands, v x v outer product
        of the velocities, and curvature of the bands (if requested). The
        shapes of those arrays are (nbands, nkpoints), (nbands, 3, 3, nkpoints)
        and (nbands, 3, 3, 3, nkpoints), where nkpoints is the total number of
        k points on the grid. If curvature is None, so will the third element
        of the tuple.
    """
    dallvec = np.vstack(equivalences)
    sallvec = mp.sharedctypes.RawArray("d", dallvec.shape[0] * 3)
    allvec = np.frombuffer(sallvec)
    allvec.shape = (-1, 3)
    dims = 2 * np.max(np.abs(dallvec), axis=0) + 1
    np.matmul(dallvec, lattvec.T, out=allvec)
    eband = np.zeros((len(coeffs), np.prod(dims)))
    vvband = np.zeros((len(coeffs), 3, 3, np.prod(dims)))
    if return_effective_mass:
        cband = np.zeros((len(coeffs), 3, 3, np.prod(dims)))
    else:
        cband = None

    # Span as many worker processes as needed, put all the bands in the queue,
    # and let them work until all the required FFTs have been computed.
    workers = []
    iqueue = mp.Queue()
    oqueue = mp.Queue()
    for iband, bandcoeff in enumerate(coeffs):
        iqueue.put((iband, bandcoeff))
    # The "None"s at the end of the queue signal the workers that there are
    # no more jobs left and they must therefore exit.
    for i in range(nworkers):
        iqueue.put(None)
    for i in range(nworkers):
        workers.append(
            mp.Process(
                target=fft_worker,
                args=(
                    equivalences,
                    sallvec,
                    dims,
                    iqueue,
                    oqueue,
                    return_effective_mass,
                ),
            )
        )
    for w in workers:
        w.start()
    # The results of the FFTs are processed as soon as they are ready.
    for r in range(len(coeffs)):
        iband, eband[iband], vvband[iband], cb = oqueue.get()
        if return_effective_mass:
            cband[iband] = cb
    for w in workers:
        w.join()
    if cband is not None:
        cband = cband.real
    return eband.real, vvband.real, cband


def fft_worker(
    equivalences, sallvec, dims, iqueue, oqueue, return_effective_mass=False
):
    """Thin wrapper around FFTev and FFTc to be used as a worker function.

    Args:
        equivalences: list of k-point equivalence classes in direct coordinates
        sallvec: Cartesian coordinates of all k points as a 1D vector stored
                    in shared memory.
        dims: upper bound on the dimensions of the k-point grid
        iqueue: input multiprocessing.Queue used to read bad indices
            and coefficients.
        oqueue: output multiprocessing.Queue where all results of the
            electronic_structure are put. Each element of the queue is a 4-tuple
            of the form (index, eband, vvband, cband), containing the band
            index, the energies, the v x v outer product and the curvatures
            if requested.
        return_effective_mass: Whether to calculate the effective mass.

    Returns:
        None. The results of the calculation are put in oqueue.
    """
    iu0 = np.triu_indices(3)
    il1 = np.tril_indices(3, -1)
    iu1 = np.triu_indices(3, 1)
    allvec = np.frombuffer(sallvec)
    allvec.shape = (-1, 3)

    while True:
        task = iqueue.get()
        if task is None:
            break
        else:
            index, bandcoeff = task
        eband, vb = FFTev(equivalences, bandcoeff, allvec, dims)
        vvband = np.zeros((3, 3, np.prod(dims)))
        effective_mass = np.zeros((3, 3, np.prod(dims)))

        vvband[iu0[0], iu0[1]] = vb[iu0[0]] * vb[iu0[1]]
        vvband[il1[0], il1[1]] = vvband[iu1[0], iu1[1]]
        if return_effective_mass:
            effective_mass[iu0] = 1 / FFTc(equivalences, bandcoeff, allvec, dims)
            effective_mass[il1] = effective_mass[iu1]
        else:
            effective_mass = None
        oqueue.put((index, eband, vvband, effective_mass))


def get_band_centers(kpoints, energies, vb_idx, efermi, tol=0.0001 * ev_to_hartree):
    band_centers = {}

    for spin, spin_energies in energies.items():
        spin_centers = []
        for i, band_energies in enumerate(spin_energies):
            if vb_idx is None:
                # handle metals
                k_idxs = np.abs(band_energies - efermi) < tol

            elif i <= vb_idx[spin]:
                k_idxs = (np.max(band_energies) - band_energies) < tol

            else:
                k_idxs = (band_energies - np.min(band_energies)) < tol

            spin_centers.append(kpoints[k_idxs])
        band_centers[spin] = spin_centers
    return band_centers
