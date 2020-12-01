import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from pymatgen import Spin
from tabulate import tabulate

from amset.constants import (
    boltzmann_au,
    coulomb_to_au,
    ev_to_hartree,
    gpa_to_au,
    m_to_bohr,
    s_to_au,
)
from amset.core.data import AmsetData, check_nbands_equal
from amset.interpolation.deformation import DeformationPotentialInterpolator
from amset.scattering.common import calculate_inverse_screening_length_sq

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

logger = logging.getLogger(__name__)


class AbstractElasticScattering(ABC):

    name: str
    required_properties: Tuple[str]

    def __init__(self, properties, doping, temperatures, nbands):
        self.properties = properties
        self.doping = doping
        self.temperatures = temperatures
        self.nbands = nbands
        self.spins = list(nbands.keys())

    @classmethod
    def from_amset_data(
        cls, materials_properties: Dict[str, Any], amset_data: AmsetData
    ):
        return cls(
            cls.get_properties(materials_properties),
            amset_data.doping,
            amset_data.temperatures,
            cls.get_nbands(amset_data),
        )

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

    def to_reference(self):
        return self.properties, self.doping, self.temperatures, self.nbands

    @classmethod
    def from_reference(cls, properties, doping, temperatures, nbands):
        return cls(properties, doping, temperatures, nbands)

    @classmethod
    def get_properties(cls, materials_properties):
        return {p: materials_properties[p] for p in cls.required_properties}

    @staticmethod
    def get_nbands(amset_data):
        return {s: len(amset_data.energies[s]) for s in amset_data.spins}


class AcousticDeformationPotentialScattering(AbstractElasticScattering):

    name = "ADP"
    required_properties = ("deformation_potential", "elastic_constant")

    def __init__(
        self,
        properties,
        doping,
        temperatures,
        nbands,
        deformation_potential,
        vb_idx,
        is_metal,
        fermi_levels,
    ):
        super().__init__(properties, doping, temperatures, nbands)
        self._prefactor = boltzmann_au * s_to_au
        self.elastic_constant = self.properties["elastic_constant"] * gpa_to_au
        self.deformation_potential = deformation_potential
        self.vb_idx = vb_idx
        self.is_metal = is_metal
        self.fermi_levels = fermi_levels

    @classmethod
    def from_amset_data(
        cls, materials_properties: Dict[str, Any], amset_data: AmsetData
    ):
        vb_idx = amset_data.vb_idx
        is_metal = amset_data.is_metal

        deformation_potential = materials_properties["deformation_potential"]
        if isinstance(deformation_potential, (str, Path)):
            deformation_potential = DeformationPotentialInterpolator.from_file(
                deformation_potential, scale=ev_to_hartree
            )
            equal = check_nbands_equal(deformation_potential, amset_data)
            if not equal:
                raise RuntimeError(
                    "Deformation potential file does not contain the correct number of"
                    " bands\nEnsure it was generated using the same energy_cutoff as "
                    "this AMSET run."
                )

        elif is_metal and isinstance(deformation_potential, tuple):
            logger.warning(
                "System is metallic but deformation potentials for both "
                "the valence and conduction bands have been set... using the "
                "valence band potential for all bands"
            )
            deformation_potential = deformation_potential[0] * ev_to_hartree

        elif is_metal:
            deformation_potential = deformation_potential * ev_to_hartree

        elif not is_metal and not isinstance(deformation_potential, tuple):
            logger.warning(
                "System is semiconducting but only one deformation "
                "potential has been set... using this potential for all bands."
            )
            deformation_potential = (
                deformation_potential * ev_to_hartree,
                deformation_potential * ev_to_hartree,
            )

        else:
            deformation_potential = (
                deformation_potential[0] * ev_to_hartree,
                deformation_potential[1] * ev_to_hartree,
            )
        return cls(
            cls.get_properties(materials_properties),
            amset_data.doping,
            amset_data.temperatures,
            cls.get_nbands(amset_data),
            deformation_potential,
            vb_idx,
            is_metal,
            amset_data.fermi_levels,
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

    def to_reference(self):
        base_reference = super().to_reference()
        if isinstance(self.deformation_potential, DeformationPotentialInterpolator):
            deformation_reference = self.deformation_potential.to_reference()
            is_interpolator = True
        else:
            deformation_reference = self.deformation_potential
            is_interpolator = False
        return base_reference + (
            deformation_reference,
            self.vb_idx,
            self.is_metal,
            self.fermi_levels,
            is_interpolator,
        )

    @classmethod
    def from_reference(
        cls,
        properties,
        doping,
        temperatures,
        nbands,
        deformation_reference,
        vb_idx,
        is_metal,
        fermi_levels,
        is_interpolator,
    ):
        if is_interpolator:
            deformation_potential = DeformationPotentialInterpolator.from_reference(
                *deformation_reference
            )
        else:
            deformation_potential = deformation_reference
        return cls(
            properties,
            doping,
            temperatures,
            nbands,
            deformation_potential,
            vb_idx,
            is_metal,
            fermi_levels,
        )


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

    def __init__(
        self,
        properties,
        doping,
        temperatures,
        nbands,
        impurity_concentration,
        inverse_screening_length_sq,
    ):
        super().__init__(properties, doping, temperatures, nbands)
        self._prefactor = impurity_concentration * s_to_au * np.pi
        self.inverse_screening_length_sq = inverse_screening_length_sq

    @classmethod
    def from_amset_data(
        cls, materials_properties: Dict[str, Any], amset_data: AmsetData
    ):
        from amset.constants import bohr_to_cm

        avg_diel = np.linalg.eigvalsh(materials_properties["static_dielectric"]).mean()
        inverse_screening_length_sq = calculate_inverse_screening_length_sq(
            amset_data, avg_diel
        )

        imp_info = []
        impurity_concentration = np.zeros(amset_data.fermi_levels.shape)
        for n, t in np.ndindex(inverse_screening_length_sq.shape):
            n_conc = np.abs(amset_data.electron_conc[n, t])
            p_conc = np.abs(amset_data.hole_conc[n, t])

            impurity_concentration[n, t] = (
                n_conc * materials_properties["donor_charge"] ** 2
                + p_conc * materials_properties["acceptor_charge"] ** 2
            )
            imp_info.append(
                (
                    amset_data.doping[n] * (1 / bohr_to_cm) ** 3,
                    amset_data.temperatures[t],
                    inverse_screening_length_sq[n, t],
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
        return cls(
            cls.get_properties(materials_properties),
            amset_data.doping,
            amset_data.temperatures,
            cls.get_nbands(amset_data),
            impurity_concentration,
            inverse_screening_length_sq,
        )

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
        "free_carrier_screening",
    )

    def __init__(
        self,
        properties,
        doping,
        temperatures,
        nbands,
        piezoelectric_constant,
        inverse_screening_length_sq,
    ):
        super().__init__(properties, doping, temperatures, nbands)
        self.piezoelectric_constant = piezoelectric_constant
        self.inverse_screening_length_sq = inverse_screening_length_sq

        self._prefactor = self.temperatures[None, :] * boltzmann_au * s_to_au
        self._shape = np.ones((len(self.doping), len(self.temperatures)))
        self.elastic_constant = self.properties["elastic_constant"] * gpa_to_au

    @classmethod
    def from_amset_data(
        cls, materials_properties: Dict[str, Any], amset_data: AmsetData
    ):
        # convert dielectric to atomic units
        shape = (len(amset_data.doping), len(amset_data.temperatures))
        e = materials_properties["piezoelectric_constant"]
        e *= coulomb_to_au / m_to_bohr ** 2  # convert to atomic units
        dielectric = materials_properties["high_frequency_dielectric"] / (4 * np.pi)
        inv_dielectric = np.linalg.inv(dielectric)

        # use h piezoelectric coefficient (Stress-Voltage)
        piezoelectric_constant = np.einsum("mn,mkl->nkl", inv_dielectric, e)

        if materials_properties["free_carrier_screening"]:
            avg_diel = np.linalg.eigvalsh(
                materials_properties["high_frequency_dielectric"]
            ).mean()
            inverse_screening_length_sq = calculate_inverse_screening_length_sq(
                amset_data, avg_diel
            )
        else:
            # fill with small value for numerical convergence
            inverse_screening_length_sq = np.full(shape, 1e-12)
        return cls(
            cls.get_properties(materials_properties),
            amset_data.doping,
            amset_data.temperatures,
            cls.get_nbands(amset_data),
            piezoelectric_constant,
            inverse_screening_length_sq,
        )

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

        return (
            factor[None, None]
            * np.ones(self._shape.shape + norm_q_sq.shape)
            / (norm_q_sq[None, None, :] + self.inverse_screening_length_sq[..., None])
        )
