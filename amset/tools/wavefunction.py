from pathlib import Path

import click


@click.command()
@click.option("-s", "--structure", default="CONTCAR", help="POSCAR/CONTCAR file")
@click.option("-p", "--potcar", default="POTCAR", help="POTCAR file")
@click.option("-w", "--wavecar", default="WAVECAR", help="WAVECAR file")
@click.option("-v", "--vasprun", default="vasprun.xml", help="vasprun.xml file")
@click.option("-d", "--directory", help="directory to look for files")
@click.option("-e", "--energy-cutoff", help="energy cutoff for finding bands")
@click.option(
    "-c", "--planewave-cutoff", default=400, help="planewave cutoff for coefficients"
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
    energy_cutoff = kwargs.pop("energy_cutoff", defaults["energy_cutoff"])
    planewave_cutoff = kwargs.pop("planewave_cutoff")

    wf = get_wavefunction(**kwargs)

    if kwargs["directory"]:
        kwargs["vasprun"] = Path(kwargs["directory"]) / "vasprun.xml"

    vr = BSVasprun(kwargs["vasprun"])
    bs = vr.get_band_structure()
    ibands = get_ibands(energy_cutoff, bs)

    click.echo("******* Getting wavefunction coefficients *******")
    coeffs = get_wavefunction_coefficients(wf, bs, iband=ibands, encut=planewave_cutoff)

    click.echo("Writing coefficients to {}".format(output))
    dump_coefficients(coeffs, wf.kpts, wf.structure, filename=output)
