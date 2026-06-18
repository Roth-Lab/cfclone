import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

from dataclasses import dataclass

from cfclone.run.postprocess import _load_results, _add_bin_cols_to_summary_df

from cfclone.plot.colours import colours


def _load_clone_data(in_file: str) -> pd.DataFrame:
    data, _, _ = _load_results(in_file)
    
    cn_t = pd.DataFrame(data['cn_t'].T, columns=data['clones'])
    
    _add_bin_cols_to_summary_df(data['bins'], cn_t)
    
    cn_t = cn_t.melt(id_vars=['chrom', 'start', 'end'], var_name='clone', value_name='total')
    
    cn_a = pd.DataFrame(data['cn_a'].T, columns=data['clones'])
    
    _add_bin_cols_to_summary_df(data['bins'], cn_a)
    
    cn_a = cn_a.melt(id_vars=['chrom', 'start', 'end'], var_name='clone', value_name='a')
    
    clones_df = cn_t.merge(cn_a)
    
    return clones_df


@dataclass
class RowPlot:
    clone: str
    df: pd.DataFrame
    yvar: str


def plot_clones(in_file: str, out_file: str, yvar: str = 'total'):
    
    clones_df = _load_clone_data(in_file)
    
    if yvar == 'total':
    
        ylims = clones_df['total'].min(), clones_df['total'].max()
        
    else:
        ylims = (0, 1)
    
    clones = clones_df['clone'].unique()[:4]
    
    rows = []
    
    for c_idx, c in enumerate(clones):
        
        rows.append(
            RowPlot(
                clone=c,
                df=clones_df.loc[clones_df['clone'] == c].copy(),
                yvar=yvar,
            )
        )
       
        
    fig = plt.figure(figsize=(6.5, 9))
    
    gs = fig.add_gridspec(nrows=len(rows))
    
    for r_idx, r in enumerate(rows):
        
        plot_data(
            df=r.df,
            fig=fig,
            grid=gs[r_idx],
            ylims=ylims,
            title="Clone {}".format(r.clone),
            yvar=r.yvar,
        )
        
    
    fig.align_labels()
    
    plt.tight_layout()
    
    plt.savefig(out_file)
    
    plt.close()
        

def plot_data(
    df: pd.DataFrame,
    fig: plt.Figure,
    grid: plt.GridSpec,
    ylims: tuple[float, float],
    yvar: str,
    title: str,
):
    
    chroms = sort_chroms(df["chrom"].unique())

    chroms_size = df["chrom"].value_counts()

    width_ratios = [chroms_size[x] for x in chroms]

    sub_grid = grid.subgridspec(
        nrows=1, ncols=len(chroms), width_ratios=width_ratios, wspace=0.05
    )

    grouped = df.groupby("chrom")

    last_chrom = chroms[-1]

    for i, chrom in enumerate(chroms):
        
        chrom_df = grouped.get_group(chrom)

        chrom_df = chrom_df.sort_values(by=["start"])

        num_bins = chrom_df.shape[0]

        chrom_df["idx"] = np.arange(num_bins)

        ax = fig.add_subplot(sub_grid[0, i])
        
        if yvar == 'total':
            
            y = chrom_df['total']
        
        else:
            
            y = chrom_df['a'] / chrom_df['total']
        
        ax.scatter(
            chrom_df["idx"],
            y,
            c=colours["orange"],
            s=1,
        )
        
        ax.set_ylim(ylims[0], ylims[1])
        
        ax.spines["top"].set_visible(False)

        ax.spines["right"].set_visible(False)
        
        if i != 0:
            ax.spines["left"].set_visible(False)

            ax.tick_params(axis='y', labelleft=False, left=False)

        else:
            ax.tick_params(axis="x", which="major")
            
        ax.set_xticks([num_bins / 2])

        ax.set_xticklabels([chrom.replace("chr", "")])
        
        ax.set_yticks(
            ticks=[int(x) for x in range(ylims[0], ylims[1] + 1)],
            labels=[str(int(x)) for x in range(ylims[0], ylims[1] + 1)],
        )
        
        if i == 7:
        
            ax.set_title(title)



# sort_chroms adapted from: https://github.com/Roth-Lab/hapclone-smk/blob/main/scripts/plot_clone_pseudobulk.py
def sort_chroms(chroms):
    numeric = []
    string = []

    if chroms[0].startswith("chr"):
        chr_prefix = True
    else:
        chr_prefix = False

    for c in chroms:
        if chr_prefix:
            c = c.replace("chr", "")
        try:
            numeric.append(int(c))
        except ValueError:
            string.append(c)

    chroms = [str(x) for x in sorted(numeric)] + list(sorted(string))

    if chr_prefix:
        chroms = ["chr{}".format(x) for x in chroms]

    return chroms