import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from cfclone.plot.colours import colours
from cfclone.plot.chrom_utils import sort_chroms


@dataclass
class RowPlot:
    sample: str
    df: pd.DataFrame
    yvar: str = 'rdr'


def plot_ctdna(ctdnas_file: str, plot_file: str, title: str | None = None, yvar: str = 'rdr'):
    df = pd.read_csv(ctdnas_file, sep='\t')
    fig = plt.figure(figsize=(16, 2))
    gs = fig.add_gridspec()
    plot_data(
        df=df,
        fig=fig,
        grid=gs[0],
        title=title,
        yvar=yvar,
    )
    fig.align_labels()
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()


def plot_data(
    df: pd.DataFrame,
    fig: plt.Figure,
    grid: plt.GridSpec,
    yvar: str,
    title: str | None = None,
    ylims: tuple[float, float] | None = None,
):
    if ylims is None:
        if yvar == "rdr":
            ylims = (df[yvar].quantile(0.025), df[yvar].quantile(0.975))
            ylims = (df[yvar].min(), df[yvar].max())
        else:
            df['baf'] = df['b'] / (df['a'] + df['b'])
            ylims = (0, 1)

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

        ax.scatter(
            chrom_df["idx"],
            chrom_df[yvar],
            c=colours["orange"],
            s=1,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i != 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", labelleft=False, left=False)
        else:
            ax.tick_params(axis="x", which="major")
        ax.set_xticks([num_bins / 2])
        ax.set_xticklabels([chrom.replace("chr", "")])
        # ax.set_yticks(
        #     ticks=[int(x) for x in range(ylims[0], ylims[1] + 1)],
        #     labels=[str(int(x)) for x in range(ylims[0], ylims[1] + 1)],
        # )

        if ylims is not None:
            ax.set_ylim(ylims[0], ylims[1])

        if title is not None and chrom == find_mid_chrom(chroms, chroms_size): 
            ax.set_title(title)



def find_mid_chrom(chroms: list[str], chroms_size: pd.Series) -> str:
    prop = 0
    for c in chroms:
        prop += chroms_size[c] / chroms_size.sum()
        if prop >= 0.5:
            return c