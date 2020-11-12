import sys
from typing import Optional, Union

import click
from click import argument, option

from amset.tools.common import echo_ibands, path_type, zero_weighted_type
from amset.util import parse_ibands

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

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
    pass


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
    from pymatgen import Structure
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
    click.echo("  - # Total deformations: {}".format(len(deformations)))

    if symprec is not None:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        spg_symbol = unicodeify_spacegroup(sga.get_space_group_symbol())
        spg_number = sga.get_space_group_number()
        click.echo("  - Spacegroup: {} ({})".format(spg_symbol, spg_number))

        deformations = list(symmetry_reduce(deformations, structure, symprec=symprec))
        click.echo("  - # Inequivalent deformations: {}".format(len(deformations)))

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
@option("-e", "--energy-cutoff", type=float, help="energy cutoff for finding bands")
@option(
    "-b",
    "--bands",
    type=str,
    help="bands to calculate the deformation for (overrides energy-cutoff)",
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
    click.echo("Reading bulk (undeformed) calculation")
    bulk_calculation = parse_calculation(bulk_folder, zero_weighted_kpoints=zwk_mode)
    bulk_structure = bulk_calculation["bandstructure"].structure

    deformation_calculations = []
    for deformation_folder in deformation_folders:
        click.echo("Reading deformation calculation in {}".format(deformation_folder))
        deformation_calculation = parse_calculation(
            deformation_folder, zero_weighted_kpoints=zwk_mode
        )
        if check_calculation(bulk_calculation, deformation_calculation):
            deformation_calculations.append(deformation_calculation)

    sga = SpacegroupAnalyzer(bulk_structure, symprec=symprec)
    spg_symbol = unicodeify_spacegroup(sga.get_space_group_symbol())
    spg_number = sga.get_space_group_number()
    click.echo("\nSpacegroup: {} ({})".format(spg_symbol, spg_number))

    strain_mapping = get_strain_mapping(bulk_structure, deformation_calculations)
    click.echo("\nFound {} strains:".format(len(strain_mapping)))
    fmt_strain = get_formatted_tensors(strain_mapping.keys())
    click.echo("  - " + "\n  - ".join(fmt_strain))

    strain_mapping = get_symmetrized_strain_mapping(
        bulk_structure, strain_mapping, symprec=symprec
    )
    click.echo("\nAfter symmetrization found {} strains:".format(len(strain_mapping)))
    fmt_strain = get_formatted_tensors(strain_mapping.keys())
    click.echo("  - " + "\n  - ".join(fmt_strain))

    if not strain_coverage_ok(list(strain_mapping.keys())):
        click.echo("\nERROR: Strains do not cover full tensor, check calculations")
        sys.exit()

    click.echo("\nCalculating deformation potentials")
    bulk_calculation["bandstructure"] = expand_bandstructure(
        bulk_calculation["bandstructure"], symprec=symprec
    )
    deformation_potentials = calculate_deformation_potentials(
        bulk_calculation, strain_mapping
    )

    print_deformation_summary(bulk_calculation["bandstructure"], deformation_potentials)

    if "bands" in kwargs and kwargs["bands"] is not None:
        ibands = parse_ibands(kwargs["bands"])
    else:
        ibands = get_ibands(energy_cutoff, bulk_calculation["bandstructure"])

    echo_ibands(ibands, bulk_calculation["bandstructure"].is_spin_polarized)
    click.echo("")
    deformation_potentials = extract_bands(deformation_potentials, ibands)

    kpoints = get_kpoints_from_bandstructure(bulk_calculation["bandstructure"])
    filename = write_deformation_potentials(
        deformation_potentials, kpoints, bulk_structure, filename=kwargs["output"]
    )
    click.echo("\nDeformation potentials written to {}".format(filename))


def check_calculation(bulk_calculation, deformation_calculation):
    from amset.deformation.potentials import get_mesh_from_band_structure

    bulk_mesh, bulk_is_shifted = get_mesh_from_band_structure(
        bulk_calculation["bandstructure"]
    )
    bulk_species = tuple(bulk_calculation["bandstructure"].structure.species)
    bulk_is_metal = bulk_calculation["bandstructure"].is_metal()

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

    if is_metal != bulk_is_metal:
        click.echo(
            "Bulk structure is {} whereas deformed structure is {}.\nSkipping "
            "deformation.".format(_metal_str[bulk_is_metal], _metal_str[is_metal])
        )
        return False
    else:
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
                    click.echo("  - spin {}:".format(spin.name))

                str_kpoint = _kpt_str.format(k=kpoint)
                click.echo("  - band: {:4d}  k-point: {}".format(b_idx, str_kpoint))
                click.echo("  - deformation potential:")
                click.echo(_tensor_str.format(*edge_deform.ravel()))
                click.echo("")
