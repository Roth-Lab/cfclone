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
    help="""Path to file where results will be written in HDF5 format.""",
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
    show_default=True,
    help="""Whether to add a a CNV profile for normal heterozyogus diploid cells.""",
)
@click.option(
    "--exec-dir",
    type=click.Path(resolve_path=True),
    help="""
    Path to directory where additional sampler info will be saved.
    """,
)
@click.option(
    "--laplace-exec-dir",
    type=click.Path(resolve_path=True),
    help="""
    Path to directory where laplace approx info info will be saved.
    """,
)
@click.option(
    "--only-normal",
    is_flag=True,
    help="""Whether to fit a model with just a normal cell population.""",
)
@click.option(
    "--outlier/--no-outlier",
    default=True,
    show_default=True,
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
    default=8,
    show_default=True,
    type=click.IntRange(1),
    help="""Number of PT chains to use for fitting.""",
)
@click.option(
    "--num-rounds",
    default=10,
    show_default=True,
    type=click.IntRange(1),
    help="""Number of rounds of PT to perform.""",
)
@click.option(
    "--pi-normal",
    default=10,
    show_default=True,
    type=click.FloatRange(0),
    help="""Dirichlet prior parameter for normal population.""",
)
@click.option(
    "--pi-tumour",
    default=0.5,
    show_default=True,
    type=click.FloatRange(0),
    help="""Dirichlet prior parameter for tumour populations.""",
)
@click.option(
    "--seed",
    default=None,
    show_default=True,
    type=click.IntRange(0),
    help="""Random seed to reproduce results. 
    If not set then a random value is used.""",
)
@click.option(
    "--sex",
    type=click.Choice(cfclone.run.SexType, case_sensitive=False),
    default="female",
    show_default=True,
    help="""Sex of sample. Determines copy number of the normal population.""",
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


@click.command(context_settings={"max_content_width": 120}, name="print-model-evidence")
@click.option(
    "-i",
    "--in-file",
    required=True,
    type=click.Path(exists=True, resolve_path=True),
    help="""Path to results from the `fit` command.""",
)
def print_model_evidence(**kwargs):
    """Print the model evidence P(X|M)."""
    cfclone.run.print_model_evidence(**kwargs)


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
    show_default=True,
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
    show_default=True,
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
    show_default=True,
    type=click.FloatRange(0, 1),
    help="""Width of HDI interval.""",
)
@click.option(
    "--normal/--no-normal",
    default=False,
    show_default=True,
    help="""Whether to include the normal population.""",
)
@click.option(
    "--renormalise/--no-renormalise",
    default=True,
    show_default=True,
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
    default=False,
    show_default=True,
    help="""Whether to compute and output model generated quantities (e.g. mu and p).""",
)
def write_samples(**kwargs):
    """Write the trace of all model parameters."""
    cfclone.run.write_samples(**kwargs)


@click.command(context_settings={"max_content_width": 120}, name="write-summary")
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
def write_summary(**kwargs):
    """Write the summary of the MCMC analysis."""
    cfclone.run.write_summary(**kwargs)


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
    show_default=True,
    type=click.FloatRange(0, 1),
    help="""Width of HDI interval.""",
)
def write_tumour_content(**kwargs):
    """Write the posterior summary for overall tumour content."""
    cfclone.run.write_tumour_content(**kwargs)


@click.command(context_settings={"max_content_width": 120}, name="write-parameter-summaries")
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
    type=click.Path(resolve_path=True, writable=True),
    help="""Path where results will be written in TSV format.""",
)
@click.option(
    "--hdi-prob",
    default=0.95,
    type=click.FloatRange(0, 1),
    help="""Width of HDI interval.""",
)
def write_parameter_summaries(**kwargs):
    """Write the posterior summary tables for mu and p model parameters."""
    cfclone.run.write_parameter_summaries(**kwargs)


@click.command(context_settings={"max_content_width": 120}, name="write-ancestral-prevalences")
@click.option(
    "-i",
    "--in-file",
    required=True,
    type=click.Path(exists=True, resolve_path=True),
    help="""Path to results from the `fit` command.""",
)
@click.option(
    "-c",
    "--clone-newick",
    required=True,
    type=click.Path(exists=True, resolve_path=True),
    help="""Path to the clonal phylogeny, in newick format.""",
)
@click.option(
    "-o",
    "--out-table",
    required=True,
    type=click.Path(resolve_path=True, writable=True),
    help="""Path where prevalence results table will be written in TSV format.""",
)
@click.option(
    "-t",
    "--tree-json",
    required=True,
    type=click.Path(resolve_path=True, writable=True),
    help="""Path where tree with computed prevalence information will be written in JSON format.""",
)
@click.option(
    "--hdi-prob",
    default=0.95,
    show_default=True,
    type=click.FloatRange(0, 1),
    help="""Width of HDI interval.""",
)
@click.option(
    "--normal/--no-normal",
    default=False,
    show_default=True,
    help="""Whether to include the normal population.""",
)
@click.option(
    "--renormalise/--no-renormalise",
    default=True,
    show_default=True,
    help="""Whether to normalise the prevalences to sum to one.""",
)
@click.option(
    "--sample-id",
    default=None,
    show_default=True,
    type=click.STRING,
    help="""Sample ID associated with the dataset.""",
)
def compute_ancestral_prevalences(**kwargs):
    """Given a clonal phylogeny, compute ancestral (and observed) clonal prevalence information."""
    cfclone.run.compute_ancestral_prevalences(**kwargs)


@click.group(name="cfclone", context_settings={"max_content_width": 140})
@click.version_option()
def main():
    pass


main.add_command(fit)
main.add_command(initialise)
main.add_command(print_model_evidence)
main.add_command(write_dominance_prob)
main.add_command(write_pairwise_ranks)
main.add_command(write_prevalence_samples)
main.add_command(write_prevalence_stats)
main.add_command(write_samples)
main.add_command(write_summary)
main.add_command(write_tumour_content)
main.add_command(write_parameter_summaries)
main.add_command(compute_ancestral_prevalences)
