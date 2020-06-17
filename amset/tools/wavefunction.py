from pathlib import Path

import click

from amset.electronic_structure.wavefunction import get_converged_encut

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"


@click.command()
@click.option("-p", "--potcar", default="POTCAR", help="POTCAR file")
@click.option("-w", "--wavecar", default="WAVECAR", help="WAVECAR file")
@click.option("-v", "--vasprun", default="vasprun.xml", help="vasprun.xml file")
@click.option("-d", "--directory", help="directory to look for files")
@click.option("-e", "--energy-cutoff", help="energy cutoff for finding bands")
@click.option(
    "-s", "--symprec", type=float, help="symprec for desymmetrizing the wavefunction"
)
@click.option(
    "-c", "--planewave-cutoff", default=None, help="planewave cutoff for coefficients"
)
@click.option("-o", "--output", default="coeffs.h5", help="output file path")
def dump_wavefunction(**kwargs):
    """Extract wavefunction coefficients from a WAVECAR"""
    from amset.electronic_structure.wavefunction import (
        get_wavefunction,
        get_wavefunction_coefficients,
        dump_coefficients,
    )
    from amset.electronic_structure.common import get_ibands
    from amset.constants import defaults
    from pymatgen.io.vasp import BSVasprun

    output = kwargs.pop("output")
    planewave_cutoff = kwargs.pop("planewave_cutoff")

    energy_cutoff = kwargs.pop("energy_cutoff")
    if not energy_cutoff:
        energy_cutoff = defaults["energy_cutoff"]

    wf = get_wavefunction(**kwargs)

    if kwargs["directory"]:
        kwargs["vasprun"] = Path(kwargs["directory"]) / "vasprun.xml"

    vr = BSVasprun(kwargs["vasprun"])
    bs = vr.get_band_structure()
    ibands = get_ibands(energy_cutoff, bs)

    if not planewave_cutoff:
        click.echo("******* Automatically choosing plane wave cutoff *******")
        planewave_cutoff = get_converged_encut(
            wf, bs, iband=ibands, max_encut=600, n_samples=2000, std_tol=0.02
        )
        click.echo("\nUsing cutoff: {} eV".format(planewave_cutoff))

    click.echo("******* Getting wavefunction coefficients *******")

    click.echo("\nIncluding:")
    for spin, spin_bands in ibands.items():
        min_b = spin_bands.min() + 1
        max_b = spin_bands.max() + 1
        click.echo("  Spin-{} bands {}—{}".format(spin.name, min_b, max_b))
    click.echo("")

    coeffs, grid = get_wavefunction_coefficients(
        wf, bs, iband=ibands, encut=planewave_cutoff
    )

    click.echo("Writing coefficients to {}".format(output))
    dump_coefficients(coeffs, wf.kpts, wf.structure, filename=output)
