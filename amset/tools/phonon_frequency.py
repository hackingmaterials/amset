import sys
from pathlib import Path
from typing import Type, Union

import click
import numpy as np

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

    click.echo("Freq & weights: ")
    for w, f in zip(weights, freqs):
        print("   {:.2f} THz   {:.2f}".format(f, w))

    click.echo("max frequency: {:.2f} THz".format(freqs.max()))
    click.echo("effective frequency: {:.2f} THz".format(effective_frequency))

    return effective_frequency


def effective_phonon_frequency_from_vasp_files(vasprun, outcar):
    eigenvalues = -vasprun.normalmode_eigenvals[::-1]
    eigenvectors = vasprun.normalmode_eigenvecs[::-1]
    born_effective_charges = outcar.born

    effective_frequency, weights, frequencies = calculate_effective_phonon_frequency(
        eigenvalues, eigenvectors, born_effective_charges
    )

    return effective_frequency, weights, frequencies


def calculate_effective_phonon_frequency(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    born_effecitve_charges: np.ndarray,
):
    # get every 3rd mode after the first 3 modes, which are acoustic
    lo_eigenvals = eigenvalues[5::3]
    lo_eigenvecs = eigenvectors[5::3]

    # get frequencies from eigenvals and convert to THz
    frequencies = 15.633302 * np.sqrt(np.abs(lo_eigenvals)) * np.sign(lo_eigenvals)

    # calculate the weight as the dipole caused by the mode
    weights = []
    for eigenvec in lo_eigenvecs:
        d = [np.dot(bx, vx) for bx, vx in zip(born_effecitve_charges, eigenvec)]
        weights.append(np.sum(np.linalg.norm(d, axis=1)))

    # normalize the weights
    weights = np.array(weights) / sum(weights)

    effective_frequency = np.sum(weights * frequencies)

    return effective_frequency, weights, frequencies


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
