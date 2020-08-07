import click

image_type = click.Choice(["pdf", "png", "svg"], case_sensitive=False)
path_type = click.Path(exists=True)
