import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from cfclone.plot.colours import colours
from cfclone.plot.chrom_utils import sort_chroms


@dataclass
class RowPlot:
    clone: str
    df: pd.DataFrame
    yvar: str


def plot_clones_from_tsv(clone_cn_file: str, plot_file: str, yvar: str = "total"):
    clones_df = pd.read_csv(clone_cn_file, sep='\t')

    if yvar == "total":
        clones_df['total'] = clones_df['cn_a'] + clones_df['cn_b']
        ylims = (clones_df['total'].min(), clones_df['total'].max())
    else:
        ylims = (0, 1)

    clones = list(clones_df["clone"].unique())
    if len(clones) > 4:
        clones = clones_df["clone"].unique()[:6]

    rows = []
    for c_idx, c in enumerate(clones):
        rows.append(
            RowPlot(
                clone=c,
                df=clones_df.loc[clones_df["clone"] == c].copy(),
                yvar=yvar,
            )
        )

    fig = plt.figure(figsize=(16, 2 * len(rows)))
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

    plt.savefig(plot_file)

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

    for i, chrom in enumerate(chroms):
        chrom_df = grouped.get_group(chrom)
        chrom_df = chrom_df.sort_values(by=["start"])
        num_bins = chrom_df.shape[0]
        chrom_df["idx"] = np.arange(num_bins)

        ax = fig.add_subplot(sub_grid[0, i])

        if yvar == "total":
            y = chrom_df["total"]
        else:
            y = chrom_df["a"] / chrom_df["total"]

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
            ax.tick_params(axis="y", labelleft=False, left=False)
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
