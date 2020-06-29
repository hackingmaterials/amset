import sys
from pathlib import Path
from typing import Type, Union
from tabulate import tabulate

import click
import numpy as np

from pymatgen import Structure
from pymatgen.core.tensors import DEFAULT_QUAD
from pymatgen.io.vasp import Outcar, Vasprun

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


@click.command()
@click.option("-v", "--vasprun", default="vasprun.xml", help="vasprun.xml file")
@click.option("-o", "--outcar", default="OUTCAR", help="OUTCAR file")
def phonon_frequency(vasprun, outcar):
    """Extract the effective phonon frequency from a VASP calculation"""
    from pymatgen.io.vasp import Outcar, Vasprun

    vasprun = get_file(vasprun, Vasprun)
    outcar = get_file(outcar, Outcar)

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
    click.echo("\neffective frequency: {:.2f} THz".format(effective_frequency))

    return effective_frequency


def effective_phonon_frequency_from_vasp_files(vasprun, outcar):
    eigenvalues = -vasprun.normalmode_eigenvals[::-1]
    eigenvectors = vasprun.normalmode_eigenvecs[::-1]

    outcar.read_lepsilon()
    born_effective_charges = outcar.born

    effective_frequency, weights, frequencies = calculate_effective_phonon_frequency(
        eigenvalues, eigenvectors, born_effective_charges, vasprun.final_structure
    )

    return effective_frequency, weights, frequencies


def calculate_effective_phonon_frequency(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    born_effecitve_charges: np.ndarray,
    structure: Structure,
):
    # get frequencies from eigenvals and convert to THz
    frequencies = 15.633302 * np.sqrt(np.abs(eigenvalues)) * np.sign(eigenvalues)

    weights = []
    for eigenvector, eigenvalue in zip(eigenvectors, eigenvalues):
        weight = get_phonon_weight(
            eigenvector, eigenvalue, born_effecitve_charges, structure
        )
        weights.append(weight)

    weights = np.array(weights) / sum(weights)
    effective_frequency = np.sum(weights * frequencies)
    return effective_frequency, weights, frequencies


def get_phonon_weight(eigenvector, eigenvalue, born_effective_charges, structure):
    # take spherical average of weight on scaled unit sphere
    directions = DEFAULT_QUAD["points"] * 0.01
    quad_weights = DEFAULT_QUAD["weights"]

    all_weights = []
    for direction in directions:
        direction_weight = []
        for atom_born, atom_vec, site in zip(
            born_effective_charges, eigenvector, structure
        ):
            v1 = np.dot(atom_born, atom_vec)
            v2 = np.dot(direction, v1)
            direction_weight.append(v2 / np.sqrt(site.specie.atomic_mass * eigenvalue))
        all_weights.append(np.abs(np.sum(direction_weight)))

    weight = np.average(all_weights * quad_weights)

    if np.isnan(weight):
        return 0
    else:
        return weight


def get_file(
    filename: Union[str, Vasprun, Outcar],
    class_type: Union[Type[Vasprun], Type[Outcar]],
) -> Union[Vasprun, Outcar]:
    if isinstance(filename, str):
        filename_gz = filename + ".gz"

        if Path(filename).exists():
            return class_type(filename)

        elif Path(filename_gz).exists():
            return class_type(filename_gz)

        else:
            print("Could not find {}. Try running with -h option".format(filename))
            sys.exit()

    elif isinstance(filename, class_type):
        return filename
