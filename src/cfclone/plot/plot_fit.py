import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sb

from matplotlib.lines import Line2D

from pathlib import Path

import yaml

from cfclone.plot.colours import colours

from dataclasses import dataclass


@dataclass
class AxisSettings:
    df: pd.DataFrame
    yvar: str
    ydata: str | None = None
    ylabel: str | None = None
    title: str | None = None
    legend: bool = False


def plot_fit(
    in_file: str,
    data_to_plot: list[str] = ["rdr", "baf"],
    out_file: str | None = None,
) -> None:

    df = pd.read_csv(in_file, sep="\t")

    plot_rows = get_plot_rows(df, data_to_plot)

    fig = plt.figure(figsize=(16, 2 * len(plot_rows)))

    grid = fig.add_gridspec(nrows=len(plot_rows), ncols=1)

    chroms = sort_chroms(df["chrom"].unique())

    chroms_size = df["chrom"].value_counts()

    width_ratios = [chroms_size[x] for x in chroms]

    for row_idx, v in enumerate(plot_rows):
        sub_grid = grid[row_idx].subgridspec(
            nrows=1,
            ncols=len(chroms),
            width_ratios=width_ratios,
        )

        plot_data(
            df=v.df,
            chroms=chroms,
            fig=fig,
            grid=sub_grid,
            title=v.title,
            yvar=v.yvar,
            ydata=v.ydata,
            ylabel=v.ylabel,
            plot_legend=v.legend,
        )

    fig.align_labels()

    grid.tight_layout(fig)

    fig.savefig(out_file)


def get_plot_rows(df: pd.DataFrame, data_to_plot: list[str]) -> list[AxisSettings]:
    plot_rows = []

    if "rdr" in data_to_plot:
        rdr_fit = AxisSettings(
            df=df,
            yvar="mu",
            ydata="rdr",
            ylabel="RDR",
            legend=True,
        )

        plot_rows.append(rdr_fit)

        rdr_res = AxisSettings(
            df=df,
            yvar="mu_residual",
            ylabel="RDR Residual",
        )

        plot_rows.append(rdr_res)

        rdr_out_bin = AxisSettings(
            df=df,
            yvar="rdr_outlier_prob",
            ylabel="RDR outlier probs",
        )

        plot_rows.append(rdr_out_bin)

    if "baf" in data_to_plot:
        baf_fit = AxisSettings(
            df=df,
            yvar="p",
            ydata="baf",
            ylabel="BAF",
        )

        plot_rows.append(baf_fit)

        baf_res = AxisSettings(
            df=df,
            yvar="p_residual",
            ylabel="BAF",
        )

        plot_rows.append(baf_res)

        baf_out_bin = AxisSettings(
            df=df,
            yvar="baf_outlier_prob",
            ylabel="BAF outlier probs",
        )

        plot_rows.append(baf_out_bin)

    return plot_rows


def plot_data(
    df: pd.DataFrame,
    chroms: list[str],
    fig: plt.Figure,
    grid: plt.GridSpec,
    yvar: str,
    ydata: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    plot_legend: bool = False,
):
    mean_col = yvar + "_mean"

    lower_col = yvar + "_lower_hdi"

    upper_col = yvar + "_upper_hdi"

    y_max = df[upper_col].max()

    y_min = df[lower_col].min()

    if ydata is not None:
        data_col = "data_" + ydata

        y_max = max(df[data_col].max(), y_max)

        y_min = min(df[data_col].min(), y_min)

    grouped = df.groupby("chrom")

    last_chrom = chroms[-1]

    for i, chrom in enumerate(chroms):
        chrom_df = grouped.get_group(chrom)

        chrom_df = chrom_df.sort_values(by=["start"])

        num_bins = chrom_df.shape[0]

        chrom_df["idx"] = np.arange(num_bins)

        ax = fig.add_subplot(grid[0, i])

        if ydata is not None:
            ax.scatter(
                chrom_df["idx"],
                chrom_df[data_col],
                c=colours["orange"],
                s=1,
            )

        ax.scatter(
            chrom_df["idx"],
            chrom_df[mean_col],
            c=colours["blue"],
            s=1,
            alpha=0.2,
        )

        ax.fill_between(
            chrom_df["idx"],
            y1=chrom_df[lower_col],
            y2=chrom_df[upper_col],
            color=colours["blue"],
            alpha=0.2,
        )

        # SETUP SPLINES

        ax.spines["left"].set_position(("outward", 10))

        ax.spines["bottom"].set_position(("outward", 10))

        ax.spines["left"].set_color("black")

        ax.spines["bottom"].set_color("black")

        ax.spines["top"].set_visible(False)

        ax.spines["right"].set_visible(False)

        ax.xaxis.tick_bottom()

        ax.yaxis.tick_left()

        ax.xaxis.grid(True, which="major", linestyle=":")

        ax.yaxis.grid(True, which="major", linestyle=":")

        sb.despine(ax=ax, offset=10)

        ax.spines["top"].set_visible(False)

        ax.spines["right"].set_visible(False)

        ax.xaxis.grid(False)

        if i != 0:
            ax.spines["left"].set_visible(False)

            ax.tick_params(axis="y", labelleft=False, left=False)

        else:
            ax.tick_params(axis="x", which="major", labelsize=12)

            ax.set_ylabel(ylabel)

        ax.set_xticks([num_bins / 2])

        ax.set_xticklabels([chrom.replace("chr", "")], fontsize=12)

        ax.set_ylim(y_min, y_max)

        if plot_legend and chrom == last_chrom:
            data = Line2D(
                [],
                [],
                label="Observed data",
                color=colours["orange"],
                marker="o",
                linestyle="",
            )

            sim = Line2D(
                [],
                [],
                label="Fit",
                color=colours["blue"],
                marker="",
                linestyle="-",
            )

            ax.legend(
                handles=[data, sim],
                loc="lower right",
                bbox_to_anchor=(0.8, 1.0),
                ncols=2,
            )

    if title is not None:
        ax = fig.add_subplot(grid[:])

        ax.axis("off")

        ax.set_title(title)


def sort_chroms(chroms: list[str]) -> list[str]:
    """sort_chroms adapted from: https://github.com/Roth-Lab/hapclone-smk/blob/main/scripts/plot_clone_pseudobulk.py"""
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
