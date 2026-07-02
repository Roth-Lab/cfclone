import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sb

from matplotlib.lines import Line2D

from collections import namedtuple

from pathlib import Path

import yaml

from cfclone.plot.colours import colours


def plot_fit(fit_file: str, out_file: str | None = None):
    pass


AxisSettings = namedtuple(
    "AxisSettings", ["df", "ydata", "ylabel", "ycolour", "title", "legend"]
)

chrom_dtype = pd.CategoricalDtype(
    categories=["chr{}".format(i) for i in list(range(1, 23))] + ["chrX"] + ["chrY"],
    ordered=True,
)

plot_settings = {
    # "figure.labelsize": 16,
    "axes.facecolor": "white",
    # "axes.titlesize": 14,
    # "axes.labelsize": 16,
    # "xtick.labelsize": 9,
    # "ytick.labelsize": 9,
    "text.usetex": True,
    "grid.color": "darkgrey",
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica",
        "Nimbus Sans",
        "Liberation Sans",
        "DejaVu Sans",
        "Arial",
    ],
    # "legend.title_fontsize": 12,
    # "legend.fontsize": 10,
    # "figure.dpi": 150,
    # "savefig.dpi": 150,
}


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


def plot_data(
    df,
    chroms,
    fig,
    grid,
    y_data,
    y_label=None,
    title=None,
    plot_legend=False,
    colour=None,
):
    mean_col = y_data + "_mean"

    lower_col = y_data + "_lb"

    upper_col = y_data + "_ub"

    real_col = y_data + "_real"

    y_max = df[upper_col].max()

    y_min = df[lower_col].min()

    y_max = max(df[real_col].max(), y_max)

    y_min = min(df[real_col].min(), y_min)

    grouped = df.groupby("chrom")

    last_chrom = chroms[-1]

    for i, chrom in enumerate(chroms):
        chrom_df = grouped.get_group(chrom)

        chrom_df = chrom_df.sort_values(by=["start"])

        num_bins = chrom_df.shape[0]

        chrom_df["idx"] = np.arange(num_bins)

        ax = fig.add_subplot(grid[0, i])

        # if y_col is not None:
        ax.scatter(
            chrom_df["idx"],
            chrom_df[real_col],
            c=colours["orange"],
            s=1,
            # alpha=0.5,
        )

        # ax.scatter(
        #     chrom_df["idx"],
        #     chrom_df[mean_col],
        #     c=colour,
        #     s=1,
        #     alpha=0.2,
        # )

        ax.fill_between(
            chrom_df["idx"],
            y1=chrom_df[lower_col],
            y2=chrom_df[upper_col],
            color=colours["grey"],
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

            ax.set_ylabel(y_label)

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
                label="Simulated data (95\% quantile)",
                color="grey",
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


def tumour_content_title(tc: dict[str, float]) -> str:
    return "Tumour content estimate\n mean: {m} HDI 95\% : [{lb}, {ub}]".format(
        m=round(tc["mean"], 3),
        lb=round(tc["lower_hdi"], 3),
        ub=round(tc["upper_hdi"], 3),
    )


def plot_meta(
    df: pd.DataFrame,
    data_to_plot: list[str] = ["rdr", "baf"],
    out_file: str | None = None,
) -> None:

    plt.rcParams.update(plot_settings)

    plot_rows = []

    for patient_idx, patient_id in enumerate(df["patient_id"].unique()):
        df_p = df.loc[df["patient_id"] == patient_id]

        for sample_idx, sample_id in enumerate(df_p["sample_id"].unique()):
            df_s = df_p.loc[(df_p["sample_id"] == sample_id)]

            if "rdr" in data_to_plot:
                tc = tumour_content_estimate(
                    patient_id=patient_id, sample_id=sample_id, restart=0
                )

                title = tumour_content_title(tc)

                rdr = AxisSettings(
                    df=df_s,
                    ydata="rdr",
                    # ylabel='RDR',
                    ylabel=r"$RDR_{i}$",
                    ycolour="grey",
                    # title="Patient {p} Sample: {s}".format(p=patient_id, s=sample_id),
                    title=title,
                    legend=True if sample_idx == 0 else False,
                )

                plot_rows.append(rdr)

            if "baf" in data_to_plot:
                baf = AxisSettings(
                    df=df_s,
                    ydata="baf",
                    # ylabel='BAF',
                    ylabel=r"$BAF_{i}$",
                    ycolour="grey",
                    # title="Patient {p} Sample: {s}".format(p=patient_id, s=sample_id),
                    title=None,
                    legend=True
                    if (sample_idx == 0) and (not "rdr" in data_to_plot)
                    else False,
                )

                plot_rows.append(baf)

    num_plot_values = len(plot_rows)

    fig = plt.figure(figsize=(16, 2 * num_plot_values))

    grid = fig.add_gridspec(num_plot_values, 1, hspace=0.1)

    chroms = sort_chroms(df["chrom"].unique())

    chroms_size = df["chrom"].value_counts()

    width_ratios = [chroms_size[x] for x in chroms]

    for i, v in enumerate(plot_rows):
        sub_grid = grid[i].subgridspec(
            nrows=1, ncols=len(chroms), width_ratios=width_ratios, wspace=0.05
        )

        plot_data(
            df=v.df,
            chroms=chroms,
            fig=fig,
            grid=sub_grid,
            title=v.title,
            y_data=v.ydata,
            y_label=v.ylabel,
            plot_legend=v.legend,
            colour=v.ycolour,
        )

    fig.align_labels()

    fig.supxlabel("Chromosome", x=0.525)

    grid.tight_layout(fig)

    fig.savefig(out_file, dpi=150, bbox_inches="tight")


def plot_wgs_sim_quantiles(config_file):

    config = yaml.safe_load(open(config_file, "r"))

    out_dir = Path(config["out_dir"])

    samples = config["patients"]

    patients = list(samples.keys())

    samples = [s for p in patients for s in samples[p]]

    df = pd.read_csv(out_dir.joinpath("data.tsv"), sep="\t").loc[
        lambda df: (df["patient_id"].isin(patients)) & (df["sample_id"].isin(samples))
    ]

    df.to_csv(out_dir.joinpath("wgs_data.tsv"))

    # plot_meta(
    #     df=df,
    #     data_to_plot=['rdr'],
    #     out_file=out_dir.joinpath('wgs_rdr.svg')
    # )

    # plot_meta(
    #     df=df,
    #     data_to_plot=['baf'],
    #     out_file=out_dir.joinpath('wgs_baf.svg'),
    # )

    # plot_meta(
    #     df=df,
    #     data_to_plot=['rdr'],
    #     out_file=out_dir.joinpath('wgs_rdr.png')
    # )

    # plot_meta(
    #     df=df,
    #     data_to_plot=['baf'],
    #     out_file=out_dir.joinpath('wgs_baf.png'),
    # )

    plot_meta(
        df=df, data_to_plot=["rdr", "baf"], out_file=out_dir.joinpath("wgs_rdr_baf.svg")
    )

    plot_meta(
        df=df, data_to_plot=["rdr", "baf"], out_file=out_dir.joinpath("wgs_rdr_baf.png")
    )
