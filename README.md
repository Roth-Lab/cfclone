# cfClone - an scWGS and cfDNA integrated analysis

A Bayesian model to perform clonal deconvolution of cfDNA given scWGS.

## Getting started

We have run cfClone on operating systems Mac OSX (Tahoe 26.6.2) and Ubuntu (20.04.6) no specialized hardware is required.
cfClone has negligable installation time as it only requires Python (3.12 or higher), Julia (1.10.10), and pixi for package management.

First clone and `cd` into this repository 

```bash
git clone --depth 1 https://github.com/Roth-Lab/cfclone.git
cd cfclone
```

to install and run cfClone download [pixi](https://pixi.prefix.dev/latest/) and run the following command

```bash
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

3. and a path to output inference to an h5 file ([see](example/results/fit.h5) for example fit).

once the data files are obtained and an output path selected, inference can be performed with default values as follows:

```bash
pixi run cfclone fit --clone-cnv-file example/data/clone_cn.tsv.gz --in-file example/data/cfdna.tsv.gz --out-file example/results/fit.h5
```

The run time should be approximately ~9 minutes.

The posterior mean and $95\%$ HDI of the tumour fraction and clone prevalences can be computed with

```bash
pixi run cfclone write-tumour-content --in-file example/results/fit.h5 --out-file example/results/tumour_content.tsv
pixi run cfclone write-prevalence-stats --in-file example/results/fit.h5 --out-file example/results/prevs.tsv
```

where the posterior summary statistics are stored at [tumour content](example/results/tumour_content.tsv) and [clone prevalence](example/results/prevs.tsv).

Additionally, see [toy example](example/example.ipynb) to model ctdna with a single clone for a subset of bins.

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

## Data input format

As briefly stated, cfClone expects two tsv files containing the cfdna data and clone copy number data.
The cfdna data contains bin-wise read depth ratios (RDR) and haplotype count data
and the clone data contains bin-wise clone haplotype specific copy number profiles.
Below we show the example input data loaded into a pandas dataframe.

```bash
>>> import pandas as pd 
>>> df_cfdna = pd.read_csv('example/data/cfdna.tsv.gz', sep='\t')
>>> df_clone_cn = pd.read_csv('example/data/clone_cn.tsv.gz', sep='\t')
>>> df_cfdna.head()
  chrom    start      end       rdr     a     b
0  chr1  1000000  1500000  0.956597  3779  1982
1  chr1  1500000  2000000  0.980388  2180  1288
2  chr1  2000000  2500000  0.977580  3952  2153
3  chr1  3000000  3500000  0.949010  4642  2280
4  chr1  3500000  4000000  0.974383  4650  2314
>>> df_clone_cn.head()
  chrom    start      end  clone  cn_a  cn_b
0  chr1        0   500000      0     4     0
1  chr1   500000  1000000      0     4     0
2  chr1  1000000  1500000      0     4     0
3  chr1  1500000  2000000      0     4     1
4  chr1  2000000  2500000      0     4     0
```

## License

cfClone
Copyright (C) 2026 Matteo Lepur, Andrew Roth, Alexandre Bouchard, Emilia Hurtado

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
