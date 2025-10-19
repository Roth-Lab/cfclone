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
    "--only-normal",
    is_flag=True,
    help="""Whether to fit a model with just a normal cell population.""",
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
@click.option(
    "--use-clone",
    multiple=True,
    type=click.STRING,
    help="""Specifies a clone profile to use. 
    Can be set multiple times to use multiple clones. 
    All other clones are excluded.
    Default is to include all clones in the input file.
    """,
)
def fit(**kwargs):
    """Fit cfClone model to data."""
    cfclone.run.fit(**kwargs)


@click.command(context_settings={"max_content_width": 120}, name="init")
def initialise(**kwargs):
    """Setup Julia environment for cfClone."""
    cfclone.run.initialise(**kwargs)


@click.command(context_settings={"max_content_width": 120}, name="write-dominance-prob")
@click.option(
    "-i",
    "--in-file",
    required=True,
    type=click.Path(exists=True, resolve_path=True),
    help="""Path to results from the `fit` command.""",
)
@click.option(
    "-o",
    "--out-file",
    required=True,
    type=click.Path(resolve_path=True),
    help="""Path where results will be written in TSV format.""",
)
@click.option(
    "--normal/--no-normal",
    default=False,
    help="""Whether to include the normal population.""",
)
def write_dominance_prob(**kwargs):
    """Write the probability a clone is the most prevalent."""
    cfclone.run.write_dominance_prob(**kwargs)


@click.command(context_settings={"max_content_width": 120}, name="write-pairwise-ranks")
@click.option(
    "-i",
    "--in-file",
    required=True,
    type=click.Path(exists=True, resolve_path=True),
    help="""Path to results from the `fit` command.""",
)
@click.option(
    "-o",
    "--out-file",
    required=True,
    type=click.Path(resolve_path=True),
    help="""Path where results will be written in TSV format.""",
)
@click.option(
    "--normal/--no-normal",
    default=False,
    help="""Whether to include the normal population.""",
)
def write_pairwise_ranks(**kwargs):
    """Write table with probability clone `i` (rows) is more prevalent than clone `j` (columns)."""
    cfclone.run.write_pairwise_ranks(**kwargs)


@click.command(context_settings={"max_content_width": 120}, name="write-prevalence-samples")
@click.option(
    "-i",
    "--in-file",
    required=True,
    type=click.Path(exists=True, resolve_path=True),
    help="""Path to results from the `fit` command.""",
)
@click.option(
    "-o",
    "--out-file",
    required=True,
    type=click.Path(resolve_path=True),
    help="""Path where results will be written in TSV format.""",
)
def write_prevalence_samples(**kwargs):
    """Write the trace of clonal prevalences."""
    cfclone.run.write_prevalence_samples(**kwargs)


@click.command(context_settings={"max_content_width": 120}, name="write-prevalence-stats")
@click.option(
    "-i",
    "--in-file",
    required=True,
    type=click.Path(exists=True, resolve_path=True),
    help="""Path to results from the `fit` command.""",
)
@click.option(
    "-o",
    "--out-file",
    required=True,
    type=click.Path(resolve_path=True),
    help="""Path where results will be written in TSV format.""",
)
@click.option(
    "--hdi-prob",
    default=0.95,
    type=click.FloatRange(0, 1),
    help="""Width of HDI interval.""",
)
@click.option(
    "--normal/--no-normal",
    default=False,
    help="""Whether to include the normal population.""",
)
@click.option(
    "--renormalise/--no-renormalise",
    default=True,
    help="""Whether to normalise the prevalences to sum to one.""",
)
def write_prevalence_stats(**kwargs):
    """Write the summary statistics of clonal prevalences."""
    cfclone.run.write_prevalence_stats(**kwargs)


@click.command(context_settings={"max_content_width": 120}, name="write-samples")
@click.option(
    "-i",
    "--in-file",
    required=True,
    type=click.Path(exists=True, resolve_path=True),
    help="""Path to results from the `fit` command.""",
)
@click.option(
    "-o",
    "--out-file",
    required=True,
    type=click.Path(resolve_path=True),
    help="""Path where results will be written in TSV format.""",
)
@click.option(
    "--compute-generated-quantities/--no-generated-quantities",
    default=True,
    help="""Whether to compute and output model generated quantities (e.g. mu and p).""",
)
def write_samples(**kwargs):
    """Write the trace of all model parameters."""
    cfclone.run.write_samples(**kwargs)


@click.command(context_settings={"max_content_width": 120}, name="write-tumour-content")
@click.option(
    "-i",
    "--in-file",
    required=True,
    type=click.Path(exists=True, resolve_path=True),
    help="""Path to results from the `fit` command.""",
)
@click.option(
    "-o",
    "--out-file",
    required=True,
    type=click.Path(resolve_path=True),
    help="""Path where results will be written in TSV format.""",
)
@click.option(
    "--hdi-prob",
    default=0.95,
    type=click.FloatRange(0, 1),
    help="""Width of HDI interval.""",
)
def write_tumour_content(**kwargs):
    """Write the posterior summary for overall tumour content."""
    cfclone.run.write_tumour_content(**kwargs)


@click.group(name="cfclone", context_settings={"max_content_width": 140})
def main():
    pass


main.add_command(fit)
main.add_command(initialise)
main.add_command(write_dominance_prob)
main.add_command(write_pairwise_ranks)
main.add_command(write_prevalence_samples)
main.add_command(write_prevalence_stats)
main.add_command(write_samples)
main.add_command(write_tumour_content)
