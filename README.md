# cfClone - an scWGS and cfDNA integrated analysis 
A Bayesian model to perform clonal deconvolution of cfDNA given scWGS.

## Getting started

First clone and `cd` into this repository 
```
git clone -depth 1 https://github.com/RothLab/cfcfclone.git
cd cfclone
```
to install and run cfClone download [pixi](https://pixi.prefix.dev/latest/) and run the following command
```
pixi run cfclone
Usage: cfclone [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  fit                          Fit cfClone model to data.
  init                         Setup Julia environment for cfClone.
  print-model-evidence         Print the model evidence P(X|M).
  write-ancestral-prevalences  Given a clonal phylogeny, compute ancestral (and observed) clonal prevalence information.
  write-dominance-prob         Write the probability a clone is the most prevalent.
  write-pairwise-ranks         Write table with probability clone `i` (rows) is more prevalent than clone `j` (columns).
  write-parameter-summaries    Write the posterior summary tables for mu and p model parameters.
  write-posterior-predictive   Write the posterior summary tables for mu and p model parameters.
  write-prevalence-samples     Write the trace of clonal prevalences.
  write-prevalence-stats       Write the summary statistics of clonal prevalences.
  write-samples                Write the trace of all model parameters.
  write-summary                Write the summary of the MCMC analysis.
  write-tumour-content         Write the posterior summary for overall tumour content.
```

The main function to perform inference is `cfclone fit` at minimum it expects

1. a tsv file that contains the bin wise read depth ratio and haplotype type counts ([see](example/data/cfdna.tsv.gz) for example)

2. a tsv file that contains the bin wise total and haplotype specific copy number matrix ([see](example/data/clone_cn.tsv.gz) for example)

3. and a path to output inference to an h5 file.

once the data files are obtained and an output path selected, inference can be performed with default values as follows:

```bash
pixi run cfclone fit --clone-cnv-file example/data/clone_cn.tsv.gz --in-file example/data/cfdna.tsv.gz --out-file example/results/fit.h5
```

The posterior mean and $95\%$ HDI of the tumour fraction and clone prevalences can be computed with

```bash
pixi run cfclone write-tumour-content --in-file example/results/fit.h5 --out-file example/results/tumour_content.tsv
pixi run cfclone write-prevalence-stats --in-file example/results/fit.h5 --out-file example/results/prevs.tsv
```

where the posterior summary statistics are stored at [tumour content](example/results/tumour_content.tsv) and [clone prevalence](example/results/prevs.tsv).

Additionally see [example](example/example.ipynb) for a toy example modelling ctdna with a single clone for a subset of bins.

### Optional values for `cfclone fit`
___

**Data Parameters**

* `--sex [female|male]`: Sets sample sex to define normal cell copy number profiles (default: `female`).
* `--use-clone TEXT`: Selects specific clone profiles to include from the input file (defaults to all).
* `--num-bins INTEGER`: Number of bins to subsample from input data for model fitting (`x >= 1`).

**Model Parameters**

* `--add-normal / --no-add-normal`: Toggles inclusion of a cell population (default: `--add-normal`).
* `--only-normal`: Restricts model to only a normal cell population.
* `--outlier / --no-outlier`: Enables or disables the outlier model component (default: `--outlier`).
* `--rdr / --no-rdr`: Enables or disables the RDR likelihood (default: `--rdr`).
* `--baf / --no-baf`: Enables or disables the BAF likelihood term (default: `--baf`).
* `--pi-normal FLOAT`: Dirichlet prior hyperparameter for the normal population fraction (default: `10`, `x >= 0`).
* `--pi-tumour FLOAT`: Dirichlet prior hyperparameter for tumour population fractions (default: `0.5`, `x >= 0`).

**Inference Parameters**

* `-t, --num-threads INTEGER`: Number of CPU threads to allocate for processing (default: `1`, `x >= 1`).
* `--num-chains INTEGER`: Number of Parallel Tempering (PT) MCMC chains (default: `8`, `x >= 1`).
* `--num-rounds INTEGER`: Number of PT sampling rounds (default: `10`, `x >= 1`).
* `--seed INTEGER`: Random seed for reproducibility (`x >= 0`).
* `--exec-dir PATH`: Directory path to write additional sampler outputs.
* `--laplace-exec-dir PATH`: Directory path to write Laplace approximation outputs.


# License
cfClone
Copyright (C) 2026 Matteo Lepur, Andrew Roth, Alexandre Bouchard, Emilia Hurtado

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
