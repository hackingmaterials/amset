from pathlib import Path

import click
import numpy as np

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


@click.command()
@click.option("-w", "--wavecar", default="WAVECAR", help="WAVECAR file")
@click.option("-v", "--vasprun", default="vasprun.xml", help="vasprun.xml file")
@click.option(
    "-p", "--potcar", default="POTCAR", help="POTCAR (only needed with --pawpyseed)"
)
@click.option("-d", "--directory", help="directory to look for files")
@click.option(
    "-e", "--energy-cutoff", type=float, help="energy cutoff for finding bands"
)
@click.option(
    "-c",
    "--planewave-cutoff",
    default=None,
    type=float,
    help="planewave cutoff for coefficients",
)
@click.option(
    "--pawpyseed", is_flag=True, help="use pawpyseed to generate coefficients"
)
@click.option("-o", "--output", default="wavefunction.h5", help="output file path")
def wavefunction(**kwargs):
    """Extract wavefunction coefficients from a WAVECAR"""
    from amset.wavefunction.io import dump_coefficients
    from amset.electronic_structure.common import get_ibands
    from amset.constants import defaults
    from pymatgen.io.vasp import BSVasprun

    output = kwargs.pop("output")
    planewave_cutoff = kwargs.pop("planewave_cutoff")
    pawpyseed = kwargs.pop("pawpyseed")

    energy_cutoff = kwargs.pop("energy_cutoff")
    if not energy_cutoff:
        energy_cutoff = defaults["energy_cutoff"]

    if kwargs["directory"]:
        vasprun_file = Path(kwargs["directory"]) / "vasprun.xml"
    else:
        vasprun_file = kwargs["vasprun"]

    try:
        vr = BSVasprun(vasprun_file)
    except FileNotFoundError:
        vr = BSVasprun(str(vasprun_file) + ".gz")

    bs = vr.get_band_structure()
    ibands = get_ibands(energy_cutoff, bs)

    click.echo("******* Getting wavefunction coefficients *******")

    click.echo("\nIncluding:")
    for spin, spin_bands in ibands.items():
        min_b = spin_bands.min() + 1
        max_b = spin_bands.max() + 1
        click.echo("  Spin-{} bands {}—{}".format(spin.name, min_b, max_b))
    click.echo("")

    if pawpyseed:
        coeffs, gpoints = _wavefunction_pawpy(bs, ibands, planewave_cutoff, **kwargs)
    else:
        coeffs, gpoints = _wavefunction_vasp(ibands, planewave_cutoff, **kwargs)

    structure = vr.final_structure
    kpoints = np.array([k.frac_coords for k in bs.kpoints])

    click.echo("Writing coefficients to {}".format(output))
    dump_coefficients(coeffs, gpoints, kpoints, structure, filename=output)


def _wavefunction_vasp(ibands, planewave_cutoff, **kwargs):
    from amset.wavefunction.vasp import (
        get_wavefunction,
        get_wavefunction_coefficients,
        get_converged_encut,
    )

    directory = kwargs.pop("directory")
    wavecar = kwargs.pop("wavecar")
    wf = get_wavefunction(wavecar=wavecar, directory=directory)

    if not planewave_cutoff:
        click.echo("******* Automatically choosing plane wave cutoff *******")
        planewave_cutoff = get_converged_encut(
            wf, iband=ibands, max_encut=600, n_samples=2000, std_tol=0.02
        )
        click.echo("\nUsing cutoff: {} eV".format(planewave_cutoff))

    return get_wavefunction_coefficients(wf, iband=ibands, encut=planewave_cutoff)


def _wavefunction_pawpy(bs, ibands, planewave_cutoff, **pawpy_kwargs):
    from amset.wavefunction.pawpyseed import (
        get_wavefunction,
        get_wavefunction_coefficients,
        get_converged_encut,
    )

    wf = get_wavefunction(**pawpy_kwargs)
    if not planewave_cutoff:
        click.echo("******* Automatically choosing plane wave cutoff *******")
        planewave_cutoff = get_converged_encut(
            wf, bs, iband=ibands, max_encut=600, n_samples=2000, std_tol=0.02
        )
        click.echo("\nUsing cutoff: {} eV".format(planewave_cutoff))

    return get_wavefunction_coefficients(wf, bs, iband=ibands, encut=planewave_cutoff)
