import re
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

    elements = vasprun.final_structure.composition.elements
    if len(set(elements)) == 1:
        raise click.ClickException(
            "This system only contains a single element and is therefore not polar.\n"
            "There will no polar optical phonon scattering and you do not need to set "
            "pop_frequency."
        )

    try:
        effective_frequency, weights, freqs = (
            effective_phonon_frequency_from_vasp_files(vasprun, outcar)
        )
        click.echo("Found NAC corrected phonon frequencies\n")
    except ValueError:
        click.echo(
            "No NAC corrected file provided. Skipping calculation requiring NAC\n"
        )

        outcar = get_file(outcar, Outcar)
        effective_frequency, weights, freqs = (
            effective_phonon_frequency_from_vasp_files_no_nac(vasprun, outcar)
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
    frequencies, eigenvectors, born_effective_charges = extract_gamma_point_data(outcar)

    effective_frequency, weights = calculate_effective_phonon_frequency(
        frequencies, eigenvectors, born_effective_charges, vasprun.final_structure
    )

    return effective_frequency, weights, frequencies


def effective_phonon_frequency_from_vasp_files_no_nac(vasprun, outcar):
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


# ------block to extract data from new outcar-----#
def parse_frequencies(line):
    """Extract frequencies from a given line."""
    return [float(value) for value in re.findall(r"[-+]?\d*\.\d+|\d+", line)]


def extract_real_parts(eigenvector_line):
    """Extract the real parts of eigenvector components from a given line."""
    components = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", eigenvector_line)
    real_parts = [
        float(components[i]) for i in range(0, len(components), 2)
    ]  # Take every other value starting from 0 (real part)
    return real_parts


def reshape_to_3x3(real_parts):
    """Reshape a flat list of real parts into a 3x3 array."""
    return np.array(real_parts).reshape(3, 3)


def extract_gamma_point_data(file_path):
    """Extract frequencies and reshaped eigenvectors for the gamma point."""
    with open(file_path, "r") as file:
        content = file.read()

    match = re.findall("PHON_BORN_CHARGES = .*", content)

    if match:
        born = np.array(list(map(float, match[0].split("=")[1].split()))).reshape(
            -1, 3, 3
        )
    else:
        raise ValueError("Could not find PHONON_BORN_CHARGES")

    content = content.split("\n")
    # Identify the phonon section
    phonon_section = []
    phonon_started = False
    for line in content:
        if "Phonons" in line:
            phonon_started = True
        if phonon_started:
            phonon_section.append(line)
        if (
            phonon_started
            and "--------------------------------------------------------------------------------"
            in line
        ):
            break

    # Initialize data containers
    q_points = []
    frequencies = []
    eigenvector_3d = []

    current_q_point = None
    current_frequencies = []
    current_real_parts = []

    for line in phonon_section:
        # Detect q-point line
        if re.match(r"\s*\d+\s+[-+]?\d+\.\d+", line):
            if current_q_point:  # Save previous q-point data
                q_points.append(current_q_point)
                frequencies.append(np.array(current_frequencies))
                eigenvector_3d.append(np.array(current_real_parts))
                current_frequencies = []
                current_real_parts = []
            current_q_point = [float(x) for x in line.split()[1:4]]
        # Parse frequencies
        elif "[THz]" in line:
            current_frequencies = parse_frequencies(line)
        # Parse eigenvectors
        elif re.match(r"\s+[-+]?\d+\.\d+", line):
            real_parts = extract_real_parts(line.strip())
            reshaped_vector = reshape_to_3x3(real_parts)
            current_real_parts.append(reshaped_vector)

    # Save the last q-point's data
    if current_q_point:
        q_points.append(current_q_point)
        frequencies.append(np.array(current_frequencies))
        eigenvector_3d.append(np.array(current_real_parts))

    # Locate gamma point
    gamma_index = None
    for i, q_point in enumerate(q_points):
        if np.allclose(q_point, [0.0, 0.0, 0.0]):
            gamma_index = i
            break

    if gamma_index is None:
        raise ValueError("Gamma point ([0.0, 0.0, 0.0]) not found in the OUTCAR file.")

    # Extract gamma point data
    gamma_frequencies = frequencies[gamma_index]
    gamma_eigenvectors_3d = np.array(eigenvector_3d[gamma_index])

    return gamma_frequencies, gamma_eigenvectors_3d, born


# --------block ends here--------#


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
