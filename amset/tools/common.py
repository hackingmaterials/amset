import click

__author__ = "Alex Ganose"
__maintainer__ = "Alex Ganose"
__email__ = "aganose@lbl.gov"

zero_weighted_type = click.Choice(["keep", "drop", "prefer"], case_sensitive=False)
image_type = click.Choice(["pdf", "png", "svg", "jpg"], case_sensitive=False)
path_type = click.Path(exists=True)


def echo_ibands(ibands, is_spin_polarized):
    if is_spin_polarized:
        click.echo("Including:")
        for spin, spin_bands in ibands.items():
            min_b = spin_bands.min() + 1
            max_b = spin_bands.max() + 1
            click.echo("  - Spin-{} bands {}—{}".format(spin.name, min_b, max_b))
    else:
        bands = list(ibands.values())[0]
        click.echo("Including bands {}—{}".format(bands.min() + 1, bands.max() + 1))
