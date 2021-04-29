from pathlib import Path

import click
import numpy as np

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

from amset.tools.common import zero_weighted_type
from amset.util import parse_ibands


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
    "-b",
    "--bands",
    type=str,
    help="bands at which to extract the wavefunction, e.g., '1:10' (overrides "
    "energy-cutoff)",
)
@click.option(
    "-z",
    "--zero-weighted-kpoints",
    help="how to process zero-weighted k-points",
    type=zero_weighted_type,
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
@click.option(
    "-v", "--vasp-type", help="set the vasp wavefunction type (std, ncl, or gam)"
)
def wave(**kwargs):
    """Extract wavefunction coefficients from a WAVECAR"""
    from pymatgen.io.vasp import BSVasprun

    from amset.constants import defaults
    from amset.electronic_structure.common import (
        get_band_structure,
        get_ibands,
        get_zero_weighted_kpoint_indices,
    )
    from amset.tools.common import echo_ibands
    from amset.wavefunction.io import write_coefficients

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

    zwk_mode = kwargs.pop("zero_weighted_kpoints")
    if not zwk_mode:
        zwk_mode = defaults["zero_weighted_kpoints"]

    bs = get_band_structure(vr, zero_weighted=zwk_mode)

    if "bands" in kwargs and kwargs["bands"] is not None:
        ibands = parse_ibands(kwargs["bands"])
    else:
        ibands = get_ibands(energy_cutoff, bs)
    ikpoints = get_zero_weighted_kpoint_indices(vr, zwk_mode)

    click.echo("******* Getting wavefunction coefficients *******\n")
    echo_ibands(ibands, bs.is_spin_polarized)
    click.echo("")

    if pawpyseed:
        coeffs, gpoints, kpoints = _wavefunction_pawpy(
            bs, ibands, planewave_cutoff, ikpoints, **kwargs
        )
    else:
        coeffs, gpoints = _wavefunction_vasp(
            ibands, planewave_cutoff, ikpoints, **kwargs
        )
        kpoints = np.array([k.frac_coords for k in bs.kpoints])

    structure = vr.final_structure

    click.echo(f"Writing coefficients to {output}")
    write_coefficients(coeffs, gpoints, kpoints, structure, filename=output)


def _wavefunction_vasp(ibands, planewave_cutoff, ikpoints, **kwargs):
    from amset.wavefunction.vasp import (
        get_converged_encut,
        get_wavefunction,
        get_wavefunction_coefficients,
    )

    directory = kwargs.pop("directory")
    wavecar = kwargs.pop("wavecar")
    vasp_type = kwargs.pop("vasp_type")
    wf = get_wavefunction(wavecar=wavecar, directory=directory, vasp_type=vasp_type)

    if not planewave_cutoff:
        click.echo("******* Automatically choosing plane wave cutoff *******")
        planewave_cutoff = get_converged_encut(
            wf, iband=ibands, ikpoints=ikpoints, n_samples=2000, std_tol=0.02
        )
        click.echo(f"\nUsing cutoff: {planewave_cutoff} eV")

    return get_wavefunction_coefficients(
        wf, iband=ibands, ikpoints=ikpoints, encut=planewave_cutoff
    )


def _wavefunction_pawpy(bs, ibands, planewave_cutoff, ikpoints, **pawpy_kwargs):
    from amset.wavefunction.pawpyseed import (
        get_converged_encut,
        get_wavefunction,
        get_wavefunction_coefficients,
    )

    wf = get_wavefunction(**pawpy_kwargs)
    if not planewave_cutoff:
        click.echo("******* Automatically choosing plane wave cutoff *******")
        planewave_cutoff = get_converged_encut(
            wf,
            bs,
            iband=ibands,
            ikpoints=ikpoints,
            max_encut=600,
            n_samples=2000,
            std_tol=0.02,
        )
        click.echo(f"\nUsing cutoff: {planewave_cutoff} eV")

    return get_wavefunction_coefficients(
        wf, bs, iband=ibands, ikpoints=ikpoints, encut=planewave_cutoff
    )
