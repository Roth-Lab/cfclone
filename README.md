# cfClone - an scWGS and cfDNA integrated analysis 
A Bayesian model to perform clonal deconvolution of cfDNA given scWGS.

# Getting started 
1. Clone and `cd` into this repository 
```
> git clone -depth 1 https://github.com/RothLab/cfcfclone.git
> cd cfclone
```
2. Download [pixi](https://pixi.prefix.dev/latest/) and run 
```
> pixi run start
✨ Pixi task (start in default): cfclone
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

## Example
See [example](example/example.ipynb) for a toy example modelling ctdna with a single clone for a subset of bins.

# License
cfClone
Copyright (C) 2026 Matteo Lepur, Andrew Roth, Alexandre Bouchard, Emilia Hurtado

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
