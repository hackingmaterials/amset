"""
This module implements a class to perform band structure interpolation using
BolzTraP2.
"""

import logging
import multiprocessing
from collections import defaultdict
from typing import Optional, Union, Tuple, List, Dict

import numpy as np
from monty.json import MSONable
from scipy.ndimage import gaussian_filter1d

from BoltzTraP2 import units, sphere, fite
from BoltzTraP2.bandlib import DOS
from spglib import spglib

from amset.electronic_structure import ElectronicStructure
from amset.utils.constants import Hartree_to_eV, m_to_cm, A_to_m, hbar, e, m_e
from pymatgen import Structure
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.bandstructure import (BandStructure,
                                                         BandStructureSymmLine)
from pymatgen.electronic_structure.dos import FermiDos, Dos
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Kpoints
from pymatgen.symmetry.bandstructure import HighSymmKpath

logger = logging.getLogger(__name__)

spin_name = {Spin.up: "spin up", Spin.down: "spin down"}


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
            for :meth:`Interpolater.get_electronic_structure`.
        soc: Whether the system was calculated using spin–orbit coupling.
        magmom: The magnetic moments for each atom.
        mommat: The band structure derivatives.
        interpolate_projections: Whether to interpolate the band structure
            projections.
    """

    def __init__(self,
                 band_structure: BandStructure,
                 num_electrons: int,
                 interpolation_factor: float = 20,
                 soc: bool = False,
                 magmom: Optional[np.ndarray] = None,
                 mommat: Optional[np.ndarray] = None,
                 interpolate_projections: bool = False):
        self._band_structure = band_structure
        self._num_electrons = num_electrons
        self._soc = soc
        self._spins = self._band_structure.bands.keys()
        self._interpolate_projections = interpolate_projections
        self._lattice_matrix = (band_structure.structure.lattice.matrix *
                                units.Angstrom)
        self._coefficients = {}
        self._projection_coefficients = defaultdict(dict)

        kpoints = np.array([k.frac_coords for k in band_structure.kpoints])
        atoms = AseAtomsAdaptor.get_atoms(band_structure.structure)

        logger.info("Getting band interpolation coefficients")

        self._equivalences = sphere.get_equivalences(
            atoms=atoms, nkpt=kpoints.shape[0] * interpolation_factor,
            magmom=magmom)

        # get the interpolation mesh used by BoltzTraP2
        self.interpolation_mesh = 2 * np.max(
            np.abs(np.vstack(self._equivalences)), axis=0) + 1

        for spin in self._spins:
            energies = band_structure.bands[spin] * units.eV
            data = DFTData(kpoints, energies, self._lattice_matrix,
                           mommat=mommat)
            self._coefficients[spin] = fite.fitde3D(data, self._equivalences)

        if self._interpolate_projections:
            logger.info("Getting projection interpolation coefficients")

            if not band_structure.projections:
                raise ValueError(
                    "interpolate_projections is True but band structure has no "
                    "projections")

            for spin in self._spins:
                for label, projection in _get_projections(
                        band_structure.projections[spin]):
                    data = DFTData(kpoints, projection, self._lattice_matrix,
                                   mommat=mommat)
                    self._projection_coefficients[spin][label] = fite.fitde3D(
                        data, self._equivalences)

    def get_electronic_structure(self,
                                 energy_cutoff: Optional[float] = None,
                                 scissor: float = None,
                                 bandgap: float = None,
                                 dos_estep: float = 0.01,
                                 dos_width: float = 0.05,
                                 symprec: float = 0.01,
                                 nworkers: int = -1
                                 ) -> ElectronicStructure:
        """Gets an ElectronicStructure object using the interpolated bands.

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
            dos_estep: The DOS energy step, where smaller numbers give more
                accuracy but are more expensive.
            dos_width: The DOS gaussian smearing width in eV.
            symprec: The symmetry tolerance used when determining the symmetry
                inequivalent k-points on which to interpolate.
            nworkers: The number of processors used to perform the
                interpolation. If set to ``-1``, the number of workers will
                be set to the number of CPU cores.

        Returns:
            The electronic structure (including energies, velocities, density of
            states and k-point information) as an ElectronicStructure object.
        """

        if self._band_structure.is_metal() and (bandgap or scissor):
            raise ValueError("{} option set but system is metallic".format(
                "bandgap" if bandgap else "scissor"))

        if not self._interpolate_projections:
            raise ValueError("Band structure projections needed to obtain full"
                             "electronic structure. Reinitialise the "
                             "interpolater with interpolate_projections=True")

        nworkers = multiprocessing.cpu_count() if nworkers == -1 else nworkers

        # determine energy cutoffs
        if energy_cutoff and self._band_structure.is_metal():
            min_e = self._band_structure.efermi - energy_cutoff
            max_e = self._band_structure.efermi + energy_cutoff
        elif energy_cutoff:
            min_e = self._band_structure.get_vbm()['energy'] - energy_cutoff
            max_e = self._band_structure.get_cbm()['energy'] + energy_cutoff
        else:
            min_e = min([self._band_structure.bands[spin].min()
                         for spin in self._spins])
            max_e = max([self._band_structure.bands[spin].max()
                         for spin in self._spins])

        logger.info("Interpolating on a {} k-point mesh".format(
            "x".join(self.interpolation_mesh)))

        energies = {}
        vvelocities = {}
        projections = defaultdict(dict)
        for spin in self._spins:
            ibands = np.any((self._band_structure.bands[spin] > min_e) &
                            (self._band_structure.bands[spin] < max_e), axis=1)

            logger.info("Interpolating {} bands {}-{}".format(
                spin_name[spin], np.where(ibands)[0].min() + 1,
                np.where(ibands)[0].max() + 1))

            energies[spin], vvelocities[spin] = fite.getBTPbands(
                self._equivalences, self._coefficients[spin][ibands],
                self._lattice_matrix, nworkers=nworkers)

            if not self._band_structure.is_metal():
                vb_idx = max(self._band_structure.get_vbm()["band_index"][spin])

                # need to know the index of the valence band after discounting
                # bands during the interpolation. As ibands is just a list of
                # True/False, we can count the number of Trues up to
                # and including the VBM to get the new number of valence bands
                new_vb_idx = sum(ibands[: vb_idx + 1])
                energies[spin] = _shift_energies(
                    energies[spin], new_vb_idx, scissor=scissor,
                    bandgap=bandgap)

            for label, proj_coeffs in self._projection_coefficients[spin]:
                projections[spin][label] = fite.getBTPbands(
                    self._equivalences, proj_coeffs[ibands],
                    self._lattice_matrix, nworkers=nworkers)[0]

        all_energies = np.vstack([energies[spin] for spin in self._spins])

        # add a few multiples of dos_width to emin and emax to account for tails
        dos_emin = np.min(energies) - dos_width * 5
        dos_emax = np.max(energies) + dos_width * 5
        npts = int(round((dos_emax - dos_emin) / dos_estep))
        emesh, densities = DOS(all_energies, erange=(dos_emin, dos_emax),
                               npts=npts)

        densities = gaussian_filter1d(densities, dos_width /
                                      (emesh[1] - emesh[0]))
        densities *= 1 if self._soc or len(self._spins) == 2 else 1
        dos = Dos(self._band_structure.efermi * units.eV, emesh * units.eV,
                  densities)

        # integrate up to Fermi level to get number of electrons
        nelect = (densities[emesh <= self._band_structure.efermi].sum() /
                  (emesh[1] - emesh[0]))
        fermi_dos = FermiDos(dos, self._band_structure.structure, nelecs=nelect)

        # get the actual k-points used in the BoltzTraP2 interpolation
        # unfortunately, BoltzTraP2 doesn't expose this information so we
        # have to get it ourselves
        ir_kpts, weights, full_kpts, ir_kpts_idx, ir_to_full_idx = _get_kpoints(
            self.interpolation_mesh, self._band_structure.structure,
            symprec=symprec, return_full_kpoints=True)

        return ElectronicStructure(
            energies, vvelocities, projections, self.interpolation_mesh,
            full_kpts, ir_kpts, weights, ir_kpts_idx, ir_to_full_idx, fermi_dos)

    def get_energies(self,
                     kpoints: Union[np.ndarray, List],
                     energy_cutoff: Optional[float] = None,
                     scissor: float = None,
                     bandgap: float = None,
                     return_velocity: bool = False,
                     return_effective_mass: bool = False,
                     return_projections: bool = False,
                     coords_are_cartesian: bool = False,
                     atomic_units: bool = False
                     ) -> Union[Dict[Spin, np.ndarray],
                                Tuple[Dict[Spin, np.ndarray]], ...]:
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
            return_effective_mass: Whether to return the band effective masses.
            return_projections: Whether to return the interpolated projections.
            coords_are_cartesian: Whether the kpoints are in cartesian or
                fractional coordinates.
            atomic_units: Return the energies, velocities, and effective_massses
                in atomic units. If False, energies will be in eV, velocities in
                cm/s, and effective masses in units of electron rest mass, m0.

        Returns:
            The band energies as dictionary of::

                {spin: energies}

            If ``return_velocity``, ``return_effective_mass`` or
            ``return_projections`` a tuple is returned, formatted as::

                (energies, Optional[velocities], Optional[effective_masses],
                 Optional[projections])

            The velocities and effective masses are given as the 1x3 trace and
            full 3x3 tensor, respectively (along cartesian directions). The
            projections are summed for each orbital type (s, p, d) across all
            atoms, and are given as::

                {spin: {orbital: projections}}
        """
        if self._band_structure.is_metal() and (bandgap or scissor):
            raise ValueError("{} option set but system is metallic".format(
                "bandgap" if bandgap else "scissor"))

        # only calculate the energies for the bands within the energy cutoff
        if energy_cutoff and self._band_structure.is_metal():
            min_e = self._band_structure.efermi - energy_cutoff
            max_e = self._band_structure.efermi + energy_cutoff
        elif energy_cutoff:
            min_e = self._band_structure.get_vbm()['energy'] - energy_cutoff
            max_e = self._band_structure.get_cbm()['energy'] + energy_cutoff
        else:
            min_e = min([self._band_structure.bands[spin].min()
                         for spin in self._spins])
            max_e = max([self._band_structure.bands[spin].max()
                         for spin in self._spins])

        if coords_are_cartesian:
            kpoints = self._band_structure.structure.lattice. \
                reciprocal_lattice.get_fractional_coords(kpoints)

        kpoints = np.asarray(kpoints)

        energies = {}
        velocities = {}
        effective_masses = {}
        projections = defaultdict(dict)
        for spin in self._spins:
            ibands = np.any((self._band_structure.bands[spin] > min_e) &
                            (self._band_structure.bands[spin] < max_e), axis=1)

            logger.info("Interpolating {} bands {}-{}".format(
                spin_name[spin], np.where(ibands)[0].min() + 1,
                np.where(ibands)[0].max() + 1))

            fitted = fite.getBands(
                kpoints, self._equivalences, self._lattice_matrix,
                self._coefficients[spin][ibands],
                curvature=return_effective_mass)

            energies[spin] = fitted[0]
            velocities[spin] = fitted[1]

            if not self._band_structure.is_metal():
                vb_idx = max(self._band_structure.get_vbm()["band_index"][spin])

                # need to know the index of the valence band after discounting
                # bands during the interpolation. As ibands is just a list of
                # True/False, we can count the number of Trues included up to
                # and including the VBM to get the new number of valence bands
                new_vb_idx = sum(ibands[: vb_idx + 1])
                energies[spin] = _shift_energies(
                    energies[spin], new_vb_idx, scissor=scissor,
                    bandgap=bandgap)

            if return_effective_mass:
                effective_masses[spin] = fitted[2]

            if not atomic_units:
                energies[spin] = energies / units.eV
                velocities[spin] = _convert_velocities(
                    velocities[spin],
                    self._band_structure.structure.lattice.matrix)

                if return_effective_mass:
                    effective_masses[spin] = _convert_effective_masses(
                        effective_masses[spin])

            if return_projections:
                for label, proj_coeffs in self._projection_coefficients[spin]:
                    projections[spin][label] = fite.getBands(
                        kpoints, self._equivalences, self._lattice_matrix,
                        proj_coeffs[ibands], curvature=False)[0]

        if not (return_velocity or return_effective_mass or
                return_projections):
            return energies

        to_return = [energies]

        if return_velocity:
            to_return.append(velocities)

        if return_effective_mass:
            to_return.append(effective_masses)

        if return_projections:
            to_return.append(projections)

        return tuple(to_return)

    def get_dos(self,
                kpoint_mesh: Union[float, int, List[int]],
                energy_cutoff: Optional[float] = None,
                scissor: Optional[float] = None,
                bandgap: Optional[float] = None,
                estep: float = 0.01,
                width: float = 0.05,
                symprec: float = 0.01
                ) -> Dos:
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

        Returns:
            The density of states.
        """
        kpoints, weights = _get_kpoints(
            kpoint_mesh, self._band_structure.structure, symprec=symprec)

        energies = self.get_energies(
            kpoints, scissor=scissor, bandgap=bandgap,
            energy_cutoff=energy_cutoff, atomic_units=False)

        energies = np.vstack([energies[spin] for spin in self._spins])

        nbands = energies.shape[0]
        nkpts = energies.shape[1]
        emin = np.min(energies) - width * 5
        emax = np.max(energies) + width * 5
        epoints = int(round((emax - emin) / estep))

        # BoltzTraP DOS kpoint_weights don't work as you'd expect so we include
        # the degeneracy manually
        all_energies = np.array([[energies[nb][nk] for nb in range(nbands)]
                                 for nk in range(nkpts)
                                 for _ in range(weights[nk])])

        emesh, dos = DOS(all_energies, erange=(emin, emax), npts=epoints)
        dos = gaussian_filter1d(dos, width / (emesh[1] - emesh[0]))
        dos *= 1 if self._soc or len(self._spins) == 2 else 1

        return Dos(self._band_structure.efermi, emesh, dos)

    def get_line_mode_band_structure(self,
                                     line_density: int = 50,
                                     energy_cutoff: Optional[float] = None,
                                     scissor: Optional[float] = None,
                                     bandgap: Optional[float] = None,
                                     symprec: float = 0.01
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

        hsk = HighSymmKpath(self._band_structure.structure,
                            symprec=symprec)
        kpoints, labels = hsk.get_kpoints(line_density=line_density,
                                          coords_are_cartesian=True)
        labels_dict = {label: kpoint for kpoint, label
                       in zip(kpoints, labels) if label != ''}

        energies = self.get_energies(
            kpoints, scissor=scissor, bandgap=bandgap, atomic_units=False,
            energy_cutoff=energy_cutoff, coords_are_cartesian=True)

        return BandStructureSymmLine(
            kpoints, energies, self._band_structure.structure.lattice,
            self._band_structure.efermi, labels_dict, coords_are_cartesian=True)


class DFTData(object):
    """DFTData object used for BoltzTraP2 interpolation.

    Note that the units used by BoltzTraP are different to those used by VASP.

    Args:
        kpoints: The k-points in fractional coordinates.
        energies: The band energies in Hartree, formatted as (nbands, nkpoints).
        lattice_matrix: The lattice matrix in Bohr^3.
        mommat: The band structure derivatives.
    """

    def __init__(self,
                 kpoints: np.ndarray,
                 energies: np.ndarray,
                 lattice_matrix: np.ndarray,
                 mommat: Optional[np.ndarray] = None):
        self.kpoints = kpoints
        self.ebands = energies
        self.lattice_matrix = lattice_matrix
        self.volume = np.abs(np.linalg.det(self.lattice_matrix))
        self.mommat = mommat

    def get_lattvec(self) -> np.ndarray:
        """Get the lattice matrix. This method is required by BoltzTraP2."""
        return self.lattice_matrix


def _shift_energies(energies: np.ndarray,
                    vb_idx: int,
                    scissor: Optional[float] = None,
                    bandgap: Optional[float] = None) -> np.ndarray:
    """Shift the band energies based on the scissor or bandgap parameter.

    Args:
        energies: The band energies, in Hartree.
        vb_idx: The band index of the valence band maximum in the energies
            array.
        scissor: The amount by which the band gap is scissored. Cannot
            be used in conjunction with the ``bandgap`` option. Has no
            effect for metallic systems.
        bandgap: Automatically adjust the band gap to this value. Cannot
            be used in conjunction with the ``scissor`` option. Has no
            effect for metallic systems.

    Returns:
        The energies, shifted according to ``scissor`` or ``bandgap``.
    """

    if scissor and bandgap:
        raise ValueError("scissor and bandgap cannot be set simultaneously")

    cb_idx = vb_idx + 1

    if bandgap:
        bandgap *= units.eV  # convert to Hartree
        interp_bandgap = energies[cb_idx:].min() - energies[:cb_idx].max()
        scissor = bandgap - interp_bandgap

    if scissor:
        scissor *= units.eV  # convert to Hartree
        energies[:cb_idx] -= scissor / 2
        energies[cb_idx:] += scissor / 2

    return energies


def _convert_velocities(velocities: np.ndarray,
                        lattice_matrix: np.ndarray) -> np.ndarray:
    """Convert velocities from atomic units to cm/s.

    TODO: Tidy this function using BoltzTraP2 units.

    Args:
        velocities: The velocities in atomic units.
        lattice_matrix: The lattice matrix in Angstrom.

    Returns:
        The velocities in cm/s.
    """
    matrix_norm = (lattice_matrix / np.linalg.norm(lattice_matrix))

    factor = Hartree_to_eV * m_to_cm * A_to_m / (hbar * 0.52917721067)
    velocities = velocities.transpose((1, 0, 2))
    velocities = np.abs(np.matmul(matrix_norm, velocities)) * factor
    velocities = velocities.transpose((0, 2, 1))

    return velocities


def _convert_effective_masses(effective_masses: np.ndarray) -> np.ndarray:
    """Convert effective masses to units of electron rest mass.

    TODO: Tidy this function using BoltzTraP2 units.

    Args:
        effective_masses: The effective masses in atomic units.

    Returns:
        The effective masses in units of electron rest masss.
    """
    factor = 0.52917721067 ** 2 * e * hbar ** 2 / (
            Hartree_to_eV * A_to_m ** 2 * m_e)
    effective_masses = effective_masses.transpose((1, 0, 2, 3))
    effective_masses = factor / effective_masses
    effective_masses = effective_masses.transpose((2, 3, 0, 1))

    return effective_masses


def _get_projections(projections: np.ndarray
                     ) -> Tuple[Tuple[str, np.ndarray], ...]:
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


def _get_kpoints(kpoint_mesh: Union[float, int, List[int]],
                 structure: Structure,
                 symprec: float = 0.01,
                 return_full_kpoints: bool = False
                 ) -> Tuple[np.ndarray, ...]:
    """Gets the symmetry inequivalent k-points from a k-point mesh.

    Follows the same process as SpacegroupAnalyzer.get_ir_reciprocal_mesh
    but is faster and allows returning of the full k-point mesh and mapping.

    Args:
        kpoint_mesh: The k-point mesh as a 1x3 array. E.g.,``[6, 6, 6]``.
            Alternatively, if a single value is provided this will be
            treated as a reciprocal density and the k-point mesh dimensions
            generated automatically.
        structure: A structure.
        symprec: Symmetry tolerance used when determining the symmetry
            inequivalent k-points on which to interpolate.
        return_full_kpoints: Whether to return the full list of k-points
            covering the entire Brillouin zone and the indices of
            inequivalent k-points.

    Returns:
        The irreducible k-points and their weights as tuple, formatted as::

            (ir_kpoints, weights)

        If return_full_kpoints, the data will be returned as::

            (ir_kpoints, weights, full_kpoints, ir_kpoints_idx, ir_to_full_idx)

        Where ``ir_kpoints_idx`` is the index of the unique irreducible k-points
        in ``full_kpoints``. ``ir_to_full_idx`` is a list of indices that can be
        used to construct the full Brillouin zone from the ir_mesh. Note the
        ir -> full conversion will only work with calculated scalar properties
        such as energy (not vector properties such as velocity).
    """
    if isinstance(kpoint_mesh, (int, float)):
        # TODO: Update this to use reciprocal length as in kgrid
        kpoint_mesh = Kpoints.automatic_density_by_vol(structure, kpoint_mesh)

    atoms = AseAtomsAdaptor().get_atoms(structure)

    mapping, grid = spglib.get_ir_reciprocal_mesh(
        kpoint_mesh, atoms, symprec=symprec)
    full_kpoints = grid / kpoint_mesh

    ir_kpoints_idx, ir_to_full_idx, weights = np.unique(
        mapping, return_inverse=True, return_counts=True)
    ir_kpoints = full_kpoints[ir_kpoints_idx]

    if return_full_kpoints:
        return ir_kpoints, weights, full_kpoints, ir_kpoints_idx, ir_to_full_idx
    else:
        return ir_kpoints, weights
