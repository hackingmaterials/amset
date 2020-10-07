from pathlib import Path

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
from pymatgen import Spin
from tabulate import tabulate

from amset.constants import (
    gpa_to_au,
    boltzmann_au,
    s_to_au,
    ev_to_hartree,
    coulomb_to_au,
    m_to_bohr,
)
from amset.core.data import AmsetData, check_nbands_equal
from amset.electronic_structure.fd import fd
from amset.interpolation.deformation import DeformationPotentialInterpolator

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)


class AbstractElasticScattering(ABC):

    name: str
    required_properties: Tuple[str]

    def __init__(self, materials_properties: Dict[str, Any], amset_data: AmsetData):
        self.properties = {p: materials_properties[p] for p in self.required_properties}
        self.doping = amset_data.doping
        self.temperatures = amset_data.temperatures
        self.nbands = {s: len(amset_data.energies[s]) for s in amset_data.spins}
        self.spins = amset_data.spins

    @abstractmethod
    def prefactor(self, spin: Spin, b_idx: int):
        pass

    @abstractmethod
    def factor(
        self,
        unit_q: np.array,
        norm_q_sq: np.ndarray,
        spin: Spin,
        band_idx: int,
        kpoint: np.ndarray,
        velocity: np.ndarray,
    ):
        pass


class AcousticDeformationPotentialScattering(AbstractElasticScattering):

    name = "ADP"
    required_properties = ("deformation_potential", "elastic_constant")

    def __init__(self, materials_properties: Dict[str, Any], amset_data: AmsetData):
        super().__init__(materials_properties, amset_data)
        self.vb_idx = amset_data.vb_idx
        self.is_metal = amset_data.is_metal
        self.fermi_levels = amset_data.fermi_levels
        self.elastic_constant = self.properties["elastic_constant"] * gpa_to_au
        self._prefactor = boltzmann_au * s_to_au

        self.deformation_potential = self.properties["deformation_potential"]
        if isinstance(self.deformation_potential, (str, Path)):
            self.deformation_potential = DeformationPotentialInterpolator.from_file(
                self.deformation_potential, scale=ev_to_hartree
            )
            equal = check_nbands_equal(self.deformation_potential, amset_data)
            if not equal:
                raise RuntimeError(
                    "Deformation potential file does not contain the correct number of"
                    " bands\nEnsure it was generated using the same energy_cutoff as "
                    "this AMSET run."
                )

        elif self.is_metal and isinstance(self.deformation_potential, tuple):
            logger.warning(
                "System is metallic but deformation potentials for both "
                "the valence and conduction bands have been set... using the "
                "valence band potential for all bands"
            )
            self.deformation_potential = self.deformation_potential[0] * ev_to_hartree

        elif self.is_metal:
            self.deformation_potential = self.deformation_potential * ev_to_hartree

        elif not self.is_metal and not isinstance(self.deformation_potential, tuple):
            logger.warning(
                "System is semiconducting but only one deformation "
                "potential has been set... using this potential for all bands."
            )
            self.deformation_potential = (
                self.deformation_potential * ev_to_hartree,
                self.deformation_potential * ev_to_hartree,
            )

        else:
            self.deformation_potential = (
                self.deformation_potential[0] * ev_to_hartree,
                self.deformation_potential[1] * ev_to_hartree,
            )

    def prefactor(self, spin: Spin, b_idx: int):
        prefactor = (
            self._prefactor
            * self.temperatures[None, :]
            * np.ones((len(self.doping), len(self.temperatures)))
        )
        return prefactor

    def factor(
        self,
        unit_q: np.array,
        norm_q_sq: np.ndarray,
        spin: Spin,
        band_idx: int,
        kpoint: np.ndarray,
        velocity: np.ndarray,
    ):
        christoffel_tensors = get_christoffel_tensors(self.elastic_constant, unit_q)
        (
            (c_trans_a, c_trans_b, c_long),
            (v_trans_a, v_trans_b, v_long),
        ) = solve_christoffel_equation(christoffel_tensors)
        if isinstance(self.deformation_potential, DeformationPotentialInterpolator):
            deform = self.deformation_potential.interpolate(spin, [band_idx], [kpoint])
            deform = np.abs(deform[0])
            deform += np.outer(velocity, velocity)  # velocity correction
            strain_long, strain_trans_a, strain_trans_b = prepare_acoustic_strains(
                unit_q, v_long, v_trans_a, v_trans_b
            )
            factor = (
                np.tensordot(strain_long, deform) ** 2 / c_long
                + np.tensordot(strain_trans_a, deform) ** 2 / c_trans_a
                + np.tensordot(strain_trans_b, deform) ** 2 / c_trans_b
            )
        elif self.is_metal:
            factor = self.deformation_potential ** 2 / c_long
        else:
            def_idx = 1 if band_idx > self.vb_idx[spin] else 0
            factor = self.deformation_potential[def_idx] ** 2 / c_long

        return factor[None, None] * np.ones(self.fermi_levels.shape + norm_q_sq.shape)


def prepare_acoustic_strains(unit_q, v_long, v_trans_a, v_trans_b):
    # orient v_long and unit_q to face the same direction
    # the einsum is just pairwise dot product along the first axis
    sign = np.sign(np.einsum("ij,ij->i", unit_q, v_long))[:, None]
    v_long *= sign
    v_trans_a *= sign
    v_trans_b *= sign

    strain_long = get_unit_strain_tensors(unit_q, v_long)
    strain_trans_a = get_unit_strain_tensors(unit_q, v_trans_a)
    strain_trans_b = get_unit_strain_tensors(unit_q, v_trans_b)
    return strain_long, strain_trans_a, strain_trans_b


def get_christoffel_tensors(elastic_constant, unit_q):
    return np.einsum("ijkl,ni,nl->njk", elastic_constant, unit_q, unit_q)


def solve_christoffel_equation(christoffel_tensors):
    eigenvalues, eigenvectors = np.linalg.eigh(christoffel_tensors)
    return eigenvalues.T, eigenvectors.transpose(2, 0, 1)


def get_unit_strain_tensors(propagation_vectors, polarization_vectors):
    return propagation_vectors[:, :, None] * polarization_vectors[:, None, :]


class IonizedImpurityScattering(AbstractElasticScattering):
    name = "IMP"
    required_properties = ("acceptor_charge", "donor_charge", "static_dielectric")

    def __init__(self, materials_properties: Dict[str, Any], amset_data: AmsetData):
        super().__init__(materials_properties, amset_data)
        from amset.constants import bohr_to_cm

        self._rlat = amset_data.structure.lattice.reciprocal_lattice.matrix

        avg_diel = np.linalg.eigvalsh(self.properties["static_dielectric"]).mean()
        self.inverse_screening_length_sq = calculate_inverse_screening_length_sq(
            amset_data, avg_diel
        )

        imp_info = []
        impurity_concentration = np.zeros(amset_data.fermi_levels.shape)
        for n, t in np.ndindex(self.inverse_screening_length_sq.shape):
            n_conc = np.abs(amset_data.electron_conc[n, t])
            p_conc = np.abs(amset_data.hole_conc[n, t])

            impurity_concentration[n, t] = (
                n_conc * self.properties["donor_charge"] ** 2
                + p_conc * self.properties["acceptor_charge"] ** 2
            )
            imp_info.append(
                (
                    amset_data.doping[n] * (1 / bohr_to_cm) ** 3,
                    amset_data.temperatures[t],
                    self.inverse_screening_length_sq[n, t],
                    impurity_concentration[n, t] * (1 / bohr_to_cm) ** 3,
                )
            )
        logger.info("Inverse screening length (β) and impurity concentration (Nᵢᵢ):")
        table = tabulate(
            imp_info,
            headers=("conc [cm⁻³]", "temp [K]", "β² [a₀⁻²]", "Nᵢᵢ [cm⁻³]"),
            numalign="right",
            stralign="center",
            floatfmt=(".2e", ".1f", ".2e", ".2e"),
        )
        logger.info(table)

        self._prefactor = impurity_concentration * s_to_au * np.pi

    def prefactor(self, spin: Spin, b_idx: int):
        # need to return prefactor with shape (nspins, ndops, ntemps, nbands)
        return self._prefactor

    def factor(
        self,
        unit_q: np.array,
        norm_q_sq: np.ndarray,
        spin: Spin,
        band_idx: int,
        kpoint: np.ndarray,
        velocity: np.ndarray,
    ):
        static_tensor = self.properties["static_dielectric"] / (4 * np.pi)
        static_diel = np.einsum("ij,ij->i", unit_q, np.dot(static_tensor, unit_q.T).T)
        diel_factor = (1 / static_diel) ** 2
        return (
            diel_factor[None, None]
            / (norm_q_sq[None, None] + self.inverse_screening_length_sq[..., None]) ** 2
        )


class PiezoelectricScattering(AbstractElasticScattering):

    name = "PIE"
    required_properties = (
        "piezoelectric_constant",
        "elastic_constant",
        "high_frequency_dielectric",
    )

    def __init__(self, materials_properties: Dict[str, Any], amset_data: AmsetData):
        super().__init__(materials_properties, amset_data)
        # convert dielectric to atomic units
        self._prefactor = self.temperatures[None, :] * boltzmann_au * s_to_au
        self._shape = np.ones((len(self.doping), len(self.temperatures)))
        self.fermi_levels = amset_data.fermi_levels
        self.elastic_constant = self.properties["elastic_constant"] * gpa_to_au
        e = self.properties["piezoelectric_constant"] * coulomb_to_au / m_to_bohr ** 2
        dielectric = self.properties["high_frequency_dielectric"] / (4 * np.pi)
        inv_dielectric = np.linalg.inv(dielectric)

        # use h piezoelectric coefficient (Stress-Voltage)
        self.piezoelectric_constant = np.einsum("mn,mkl->nkl", inv_dielectric, e)

    def prefactor(self, spin: Spin, b_idx: int):
        # need to return prefactor with shape (ndops, ntemps)
        return self._prefactor * self._shape

    def factor(
        self,
        unit_q: np.array,
        norm_q_sq: np.ndarray,
        spin: Spin,
        band_idx: int,
        kpoint: np.ndarray,
        velocity: np.ndarray,
    ):
        christoffel_tensors = get_christoffel_tensors(self.elastic_constant, unit_q)
        (
            (c_trans_a, c_trans_b, c_long),
            (v_trans_a, v_trans_b, v_long),
        ) = solve_christoffel_equation(christoffel_tensors)

        strain_long, strain_trans_a, strain_trans_b = prepare_acoustic_strains(
            unit_q, v_long, v_trans_a, v_trans_b
        )
        qh = np.einsum("ijk,nj->nik", self.piezoelectric_constant, unit_q)

        # einsum is double dot product along first axis
        factor = (
            np.einsum("nij,nij->n", strain_long, qh) ** 2 / c_long
            + np.einsum("nij,nij->n", strain_trans_a, qh) ** 2 / c_trans_a
            + np.einsum("nij,nij->n", strain_trans_b, qh) ** 2 / c_trans_b
        )

        # add small number for numerical convergence
        return (
            factor[None, None]
            * np.ones(self.fermi_levels.shape + norm_q_sq.shape)
            / (norm_q_sq[None, None, :] + 1e-12)
        )


def calculate_inverse_screening_length_sq(amset_data, static_dielectric):
    inverse_screening_length_sq = np.zeros(amset_data.fermi_levels.shape)

    tdos = amset_data.dos.tdos
    energies = amset_data.dos.energies
    fermi_levels = amset_data.fermi_levels
    vol = amset_data.structure.volume

    for n, t in np.ndindex(inverse_screening_length_sq.shape):
        ef = fermi_levels[n, t]
        temp = amset_data.temperatures[t]
        f = fd(energies, ef, temp * boltzmann_au)
        integral = np.trapz(tdos * f * (1 - f), x=energies)
        inverse_screening_length_sq[n, t] = (
            integral * 4 * np.pi / (static_dielectric * boltzmann_au * temp * vol)
        )

    return inverse_screening_length_sq
