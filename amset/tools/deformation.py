import sys
from typing import Optional, Union

import click
from click import argument, option

from amset.electronic_structure.symmetry import reciprocal_lattice_match
from amset.tools.common import echo_ibands, path_type, zero_weighted_type
from amset.util import parse_ibands

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

from pymatgen.electronic_structure.core import Spin

_symprec = 0.01  # redefine symprec to avoid loading constants from file
_metal_str = {True: "metallic", False: "semiconducting"}
_kpt_str = "[{k[0]:6.2f} {k[1]:6.2f} {k[2]:6.2f}  ]"
_tensor_str = """    [[{:6.2f} {:6.2f} {:6.2f}]
     [{:6.2f} {:6.2f} {:6.2f}]
     [{:6.2f} {:6.2f} {:6.2f}]]"""


def _parse_symprec(var: Optional[Union[str, float, int]]):
    if var is None:
        return _symprec
    if isinstance(var, float):
        return var
    if "N" in var:
        return None
    return float(var)


@click.group()
def deform():
    """
    Tools for calculating acoustic deformation potentials.
    """


@deform.command()
@option("-f", "--filename", default="POSCAR", help="path to input structure file")
@option(
    "-d", "--distance", type=float, default=0.005, help="fractional magnitude of strain"
)
@option(
    "-s",
    "--symprec",
    help="symmetry precision for reducing deformations (use 'N' for no symmetry)",
)
@option("--directory", type=path_type, help="file output directory")
def create(**kwargs):
    """
    Generate deformed structures for calculating deformation potentials.
    """
    from pymatgen.core.structure import Structure
    from pymatgen.core.tensors import symmetry_reduce
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.util.string import unicodeify_spacegroup

    from amset.deformation.common import get_formatted_tensors
    from amset.deformation.generation import get_deformations, get_deformed_structures
    from amset.deformation.io import write_deformed_poscars

    symprec = _parse_symprec(kwargs["symprec"])

    structure = Structure.from_file(kwargs["filename"])

    click.echo("Generating deformations:")
    click.echo("  - Strain distance: {:g}".format(kwargs["distance"]))

    deformations = get_deformations(kwargs["distance"])
    click.echo(f"  - # Total deformations: {len(deformations)}")

    if symprec is not None:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        spg_symbol = unicodeify_spacegroup(sga.get_space_group_symbol())
        spg_number = sga.get_space_group_number()
        click.echo(f"  - Spacegroup: {spg_symbol} ({spg_number})")

        deformations = list(symmetry_reduce(deformations, structure, symprec=symprec))
        click.echo(f"  - # Inequivalent deformations: {len(deformations)}")

    click.echo("\nDeformations:")
    click.echo("  - " + "\n  - ".join(get_formatted_tensors(deformations)))

    deformed_structures = get_deformed_structures(structure, deformations)

    write_deformed_poscars(deformed_structures, directory=kwargs["directory"])
    click.echo("\nDeformed structures have been created")


@deform.command()
@argument("bulk-folder", type=path_type)
@argument("deformation-folders", nargs=-1, type=path_type)
@option(
    "-s",
    "--symprec",
    help="symmetry precision for reducing deformations (use 'N' for no symmetry)",
)
@option(
    "-d",
    "--symprec-deformation",
    default=_symprec / 100,
    help="symmetry precision for deformations structures (should be ~VASP SYMPREC)",
)
@option("-e", "--energy-cutoff", type=float, help="energy cutoff for finding bands")
@option(
    "-b",
    "--bands",
    type=str,
    help="bands to calculate the deformation for, e.g., '1:10' (overrides "
    "energy-cutoff)",
)
@option(
    "-z",
    "--zero-weighted-kpoints",
    help="how to process zero-weighted k-points",
    type=zero_weighted_type,
)
@option("-o", "--output", default="deformation.h5", help="output file path")
def read(bulk_folder, deformation_folders, **kwargs):
    """
    Read deformation calculations and extract deformation potentials.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.util.string import unicodeify_spacegroup

    from amset.constants import defaults
    from amset.deformation.common import get_formatted_tensors
    from amset.deformation.io import parse_calculation, write_deformation_potentials
    from amset.deformation.potentials import (
        calculate_deformation_potentials,
        extract_bands,
        get_strain_mapping,
        get_symmetrized_strain_mapping,
        strain_coverage_ok,
    )
    from amset.electronic_structure.common import get_ibands
    from amset.electronic_structure.kpoints import get_kpoints_from_bandstructure
    from amset.electronic_structure.symmetry import expand_bandstructure

    energy_cutoff = kwargs.pop("energy_cutoff")
    if not energy_cutoff:
        energy_cutoff = defaults["energy_cutoff"]

    zwk_mode = kwargs.pop("zero_weighted_kpoints")
    if not zwk_mode:
        zwk_mode = defaults["zero_weighted_kpoints"]

    symprec = _parse_symprec(kwargs["symprec"])
    symprec_deformation = kwargs["symprec_deformation"]

    click.echo("Reading bulk (undeformed) calculation")
    bulk_calculation = parse_calculation(bulk_folder, zero_weighted_kpoints=zwk_mode)
    bulk_structure = bulk_calculation["bandstructure"].structure

    deformation_calculations = []
    for deformation_folder in deformation_folders:
        click.echo(f"Reading deformation calculation in {deformation_folder}")
        deformation_calculation = parse_calculation(
            deformation_folder, zero_weighted_kpoints=zwk_mode
        )
        deformation_calculation = check_calculation(
            bulk_calculation, deformation_calculation
        )
        if deformation_calculation is not False:
            deformation_calculations.append(deformation_calculation)

    if symprec is not None:
        sga = SpacegroupAnalyzer(bulk_structure, symprec=symprec)
        spg_symbol = unicodeify_spacegroup(sga.get_space_group_symbol())
        spg_number = sga.get_space_group_number()
        click.echo(f"\nSpacegroup: {spg_symbol} ({spg_number})")

        lattice_match = reciprocal_lattice_match(
            bulk_calculation["bandstructure"], symprec=symprec
        )
        if not lattice_match:
            click.echo(
                "\nWARNING: Reciprocal lattice and k-lattice belong to different\n"
                "         class of lattices. Often results are still useful but\n"
                "         it is recommended to regenerate deformations without\n"
                "         symmetry using: amset deform create --symprec N"
            )

    strain_mapping = get_strain_mapping(bulk_structure, deformation_calculations)
    click.echo(f"\nFound {len(strain_mapping)} strains:")
    fmt_strain = get_formatted_tensors(strain_mapping.keys())
    click.echo("  - " + "\n  - ".join(fmt_strain))

    if symprec is not None:
        bulk_calculation["bandstructure"] = expand_bandstructure(
            bulk_calculation["bandstructure"], symprec=symprec
        )
        strain_mapping = get_symmetrized_strain_mapping(
            bulk_structure,
            strain_mapping,
            symprec=symprec,
            symprec_deformation=symprec_deformation,
        )
        click.echo(f"\nAfter symmetrization found {len(strain_mapping)} strains:")
        fmt_strain = get_formatted_tensors(strain_mapping.keys())
        click.echo("  - " + "\n  - ".join(fmt_strain))

    if not strain_coverage_ok(list(strain_mapping.keys())):
        click.echo("\nERROR: Strains do not cover full tensor, check calculations")
        sys.exit()

    strain_bs = [c["bandstructure"] for c in strain_mapping.values()]
    if not bz_coverage_ok([bulk_calculation["bandstructure"]] + strain_bs):
        click.echo(
            "\nERROR: one or more k-point meshes do not cover the full BZ. If using\n"
            "       --symprec N please ensure that the VASP calculations were \n"
            "       performed using ISYM = -1. Alternatively, set --symprec to a \n"
            "       number, e.g. 0.1."
        )
        sys.exit()

    click.echo("\nCalculating deformation potentials")
    deformation_potentials = calculate_deformation_potentials(
        bulk_calculation, strain_mapping
    )

    print_deformation_summary(bulk_calculation["bandstructure"], deformation_potentials)

    if "bands" in kwargs and kwargs["bands"] is not None:
        ibands = parse_ibands(kwargs["bands"])
    else:
        ibands = get_ibands(energy_cutoff, bulk_calculation["bandstructure"])

    echo_ibands(ibands, bulk_calculation["bandstructure"].is_spin_polarized)
    deformation_potentials = extract_bands(deformation_potentials, ibands)

    kpoints = get_kpoints_from_bandstructure(bulk_calculation["bandstructure"])
    filename = write_deformation_potentials(
        deformation_potentials, kpoints, bulk_structure, filename=kwargs["output"]
    )
    click.echo(f"\nDeformation potentials written to {filename}")


@deform.command()
@argument("deformation-file", type=path_type)
@option("-o", "--output", default="deformation_soc.h5", help="output file path")
def to_soc(deformation_file, output):
    """
    Double up all bands in the deformation file to simulate a SOC calculation.
    """
    import numpy as np

    from amset.deformation.io import (
        load_deformation_potentials,
        write_deformation_potentials,
    )

    deformation_potentials, kpoints, structure = load_deformation_potentials(
        deformation_file
    )

    if len(deformation_potentials) == 2:
        # spin polarized, order the bands by max eigenvalue
        click.echo("ERROR: Spin polarized systems are not supported.")
        sys.exit()

    new_deformation_potentials = {
        Spin.up: np.repeat(deformation_potentials[Spin.up], 2, axis=0)
    }

    filename = write_deformation_potentials(
        new_deformation_potentials, kpoints, structure, filename=output
    )
    click.echo(f"\nFake SOC deformation potentials written to {filename}")


def check_calculation(bulk_calculation, deformation_calculation):
    from amset.deformation.potentials import get_mesh_from_band_structure

    bulk_mesh, bulk_is_shifted = get_mesh_from_band_structure(
        bulk_calculation["bandstructure"]
    )
    bulk_species = tuple(bulk_calculation["bandstructure"].structure.species)
    bulk_is_metal = bulk_calculation["bandstructure"].is_metal()
    bulk_nbands = bulk_calculation["bandstructure"].nb_bands

    mesh, is_shifted = get_mesh_from_band_structure(
        deformation_calculation["bandstructure"]
    )
    if mesh != bulk_mesh or is_shifted != bulk_is_shifted:
        raise RuntimeError(
            "Calculations were not performed using the same k-point "
            "mesh\n{} != {}".format(mesh, bulk_mesh)
        )

    species = tuple(deformation_calculation["bandstructure"].structure.species)
    if species != bulk_species:
        raise RuntimeError("Calculations were performed using different structures")
    is_metal = deformation_calculation["bandstructure"].is_metal()

    nbands = deformation_calculation["bandstructure"].nb_bands
    if bulk_nbands > nbands:
        click.echo(
            "Bulk calculation has more bands than deformation. Skipping deformation ..."
        )
        return False
    elif bulk_nbands < nbands:
        click.echo(
            f"Deformation calculation has more bands than bulk "
            f"({nbands} > {bulk_nbands}).\nTrimming excess bands."
        )
        deformation_calculation["bandstructure"].bands = {
            spin: deformation_calculation["bandstructure"].bands[spin][: len(bands)]
            for spin, bands in bulk_calculation["bandstructure"].bands.items()
        }

    if is_metal != bulk_is_metal:
        click.echo(
            f"Bulk structure is {_metal_str[bulk_is_metal]} whereas deformed structure "
            f"is {_metal_str[is_metal]}.\nSkipping deformation."
        )
        return False

    return deformation_calculation


def bz_coverage_ok(bandstructures):
    import numpy as np

    from amset.deformation.potentials import get_mesh_from_band_structure

    for bandstructure in bandstructures:
        bulk_mesh, _ = get_mesh_from_band_structure(bandstructure)
        nkpoints = np.product(bulk_mesh)
        if nkpoints != len(bandstructure.kpoints):
            return False
    return True


def print_deformation_summary(bandstructure, deformation_potentials):
    if bandstructure.is_metal():
        return
    else:
        cbm = bandstructure.get_cbm()
        vbm = bandstructure.get_vbm()
        click.echo("\nValence band maximum:")
        print_band_edge_information(bandstructure, vbm, deformation_potentials)

        click.echo("Conduction band minimum:")
        print_band_edge_information(bandstructure, cbm, deformation_potentials)


def print_band_edge_information(bandstructure, band_edge, deformation_potentials):
    for spin, spin_band_idxs in band_edge["band_index"].items():
        for b_idx in spin_band_idxs:
            for k_idx in band_edge["kpoint_index"]:
                kpoint = bandstructure.kpoints[k_idx].frac_coords
                edge_deform = deformation_potentials[spin][b_idx, k_idx]

                if len(deformation_potentials) == 2:
                    click.echo(f"  - spin {spin.name}:")

                str_kpoint = _kpt_str.format(k=kpoint)
                click.echo(f"  - band: {b_idx + 1:4d}  k-point: {str_kpoint}")
                click.echo("  - deformation potential:")
                click.echo(_tensor_str.format(*edge_deform.ravel()))
                click.echo("")
