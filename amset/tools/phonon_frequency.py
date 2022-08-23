import sys
from pathlib import Path

import click
import numpy as np

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


@click.command()
@click.option("-v", "--vasprun", default="vasprun.xml", help="vasprun.xml file")
@click.option("-o", "--outcar", default="OUTCAR", help="OUTCAR file")
def phonon_frequency(vasprun, outcar):
    """Extract the effective phonon frequency from a VASP calculation"""
    from pymatgen.io.vasp import Outcar, Vasprun
    from tabulate import tabulate

    vasprun = get_file(vasprun, Vasprun)
    outcar = get_file(outcar, Outcar)

    elements = vasprun.final_structure.composition.elements
    if len(set(elements)) == 1:
        raise click.ClickException(
            "This system only contains a single element and is therefore not polar.\n"
            "There will no polar optical phonon scattering and you do not need to set "
            "pop_frequency."
        )

    effective_frequency, weights, freqs = effective_phonon_frequency_from_vasp_files(
        vasprun, outcar
    )

    table = tabulate(
        list(zip(freqs, weights)),
        headers=("Frequency", "Weight"),
        numalign="right",
        stralign="center",
        floatfmt=(".2f", ".2f"),
    )
    click.echo(table)
    click.echo(f"\npop_frequency: {effective_frequency:.2f} THz")

    return effective_frequency


def effective_phonon_frequency_from_vasp_files(vasprun, outcar):
    eigenvalues = -vasprun.normalmode_eigenvals[::-1]
    eigenvectors = vasprun.normalmode_eigenvecs[::-1]

    # get frequencies from eigenvals (and convert to THz for VASP 5)
    major_version = int(vasprun.vasp_version.split(".")[0])
    frequencies = np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues)
    if major_version < 6:
        # convert to THz
        frequencies *= 15.633302

    outcar.read_lepsilon()
    born_effective_charges = outcar.born

    effective_frequency, weights = calculate_effective_phonon_frequency(
        frequencies, eigenvectors, born_effective_charges, vasprun.final_structure
    )

    return effective_frequency, weights, frequencies


def calculate_effective_phonon_frequency(
    frequencies: np.ndarray,
    eigenvectors: np.ndarray,
    born_effecitve_charges: np.ndarray,
    structure,
):
    # frequencies should be in THz
    weights = []
    for eigenvector, frequency in zip(eigenvectors, frequencies):
        weight = get_phonon_weight(
            eigenvector, frequency, born_effecitve_charges, structure
        )
        weights.append(weight)

    weights = np.array(weights) / sum(weights)
    effective_frequency = np.sum(weights * frequencies)
    return effective_frequency, weights


def get_phonon_weight(eigenvector, frequency, born_effective_charges, structure):
    from pymatgen.core.tensors import DEFAULT_QUAD

    # take spherical average of weight on scaled unit sphere
    directions = DEFAULT_QUAD["points"] * 0.01
    quad_weights = DEFAULT_QUAD["weights"]

    # precalculate Z*.e(q) / sqrt(M*omega) for each atom; this only needs to be
    # computed once and is the same for every q direction
    zes = []
    for atom_born, atom_vec, site in zip(
        born_effective_charges, eigenvector, structure
    ):
        ze = np.dot(atom_born, atom_vec) / np.sqrt(site.specie.atomic_mass * frequency)
        zes.append(ze)

    all_weights = []
    for direction in directions:
        # Calculate q.Z*.e(q) / sqrt(M*omega) for each atom
        qze = np.dot(zes, direction)

        # sum across all atoms
        all_weights.append(np.abs(np.sum(qze)))

    weight = np.average(all_weights * quad_weights)

    if np.isnan(weight):
        return 0
    else:
        return weight


def get_file(filename, class_type):
    if isinstance(filename, str):
        filename_gz = filename + ".gz"

        if Path(filename).exists():
            return class_type(filename)

        elif Path(filename_gz).exists():
            return class_type(filename_gz)

        else:
            print(f"Could not find {filename}. Try running with -h option")
            sys.exit()

    elif isinstance(filename, class_type):
        return filename
