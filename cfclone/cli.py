import click

import cfclone.run


@click.command(context_settings={"max_content_width": 120}, name="fit")
@click.option(
    "-c",
    "--clone-cnv-file",
    required=True,
    type=click.Path(exists=True, resolve_path=True),
    help="""Path to file clone CNV profiles.""",
)
@click.option(
    "-i",
    "--in-file",
    required=True,
    type=click.Path(exists=True, resolve_path=True),
    help="""Path to file with bulk WGS data for clonal deconvolution.""",
)
@click.option(
    "-o",
    "--out-file",
    required=True,
    type=click.Path(resolve_path=True),
    help="""Path to where results will be written in HDF5 format.""",
)
@click.option(
    "-t",
    "--num-threads",
    default=1,
    show_default=True,
    type=click.IntRange(1),
    help="""Number of threads to use.""",
)
@click.option(
    "--add-normal/--no-add-normal",
    default=True,
    help="""Whether to add a a CNV profile for normal heterozyogus diploid cells.""",
)
@click.option(
    "--outlier/--no-outlier",
    default=True,
    help="""Whether to use outlier model.""",
)
@click.option(
    "--num-bins",
    default=None,
    show_default=True,
    type=click.IntRange(1),
    help="""Number of bins to subsample for fitting.""",
)
@click.option(
    "--num-chains",
    default=12,
    show_default=True,
    type=click.IntRange(1),
    help="""Number of PT chains to use for fitting.""",
)
@click.option(
    "--num-chains-vi",
    default=12,
    show_default=True,
    type=click.IntRange(1),
    help="""Number of chains for the variational reference.""",
)
@click.option(
    "--num-rounds",
    default=10,
    show_default=True,
    type=click.IntRange(1),
    help="""Number of rounds of PT to perform.""",
)
def fit(**kwargs):
    """Fit cfClone model to data."""
    cfclone.run.fit(**kwargs)


@click.command(context_settings={"max_content_width": 120}, name="init")
def initialise(**kwargs):
    """Setup Julia environment for cfClone."""
    cfclone.run.initialise(**kwargs)


@click.group(name="cfclone")
def main():
    pass


main.add_command(fit)
main.add_command(initialise)
