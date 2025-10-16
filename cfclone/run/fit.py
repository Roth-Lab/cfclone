import importlib.resources
import os

import numpy as np
import pandas as pd
import random

from cfclone.inference import run_inference
from cfclone.julia import setup_julia_module
from cfclone.models import get_model


def fit(
    clone_cnv_file,
    in_file,
    out_file,
    add_normal=True,
    num_bins=None,
    num_chains=12,
    num_chains_vi=5,
    num_rounds=10,
    num_threads=1,
    outlier=False,
):
    bins, clones, data = load_data(clone_cnv_file, in_file, add_normal=add_normal, num_bins=num_bins)

    print(clones)

    os.environ["TBB_CXX_TYPE"] = "gcc"
    os.environ["TBB_INTERFACE_NEW"] = "new"
    os.environ["STAN_THREADS"] = "true"
    os.environ["STAN_NUM_THREADS"] = f"{num_threads}"

    jl = setup_julia_module()

    model = get_model(jl, data, use_outlier=outlier)

    pt = run_inference(
        jl,
        model,
        num_chains=num_chains,
        num_chains_vi=num_chains_vi,
        num_rounds=num_rounds,
    )

    names = [str(x).split(":")[-1] for x in list(jl.sample_names(pt))]

    results = jl.sample_array(pt).to_numpy()

    out_df = []

    for i in range(results.shape[2]):
        chain_results = results[:, :, i]

        chain_df = pd.DataFrame(chain_results, columns=names)

        chain_df.insert(0, "chain", i)

        out_df.append(chain_df)

    out_df = pd.concat(out_df)

    rho_map = {}

    for col in out_df.columns:
        if col.startswith("rho"):
            i = int(col.split(".")[1]) - 1

            clone = clones[i]

            rho_map[col] = "rho_{}".format(clone)

    out_df = out_df.rename(columns=rho_map)

    out_df.to_csv(out_file, compression="gzip", index=False, sep="\t")


def load_data(clone_cnv_file, in_file, add_normal=True, num_bins=None):

    df = pd.read_csv(in_file, sep="\t")

    _add_bin_name_col(df)

    clone_df = pd.read_csv(clone_cnv_file, sep="\t")

    _add_bin_name_col(clone_df)

    # Ensure the same set of bins is used and data is aligned
    bins = pd.merge(
        df[["bin_name"]].drop_duplicates(), clone_df[["bin_name"]].drop_duplicates(), on="bin_name", how="inner"
    )["bin_name"]

    if num_bins is not None:
        bins = random.sample(list(bins), num_bins)

    a = df.set_index("bin_name").loc[bins, "a"]

    b = df.set_index("bin_name").loc[bins, "b"]

    d = a + b

    rdr = df.set_index("bin_name").loc[bins, "rdr"]

    cn_a = clone_df.pivot(index="clone", columns="bin_name", values="cn_a")[bins]

    cn_b = clone_df.pivot(index="clone", columns="bin_name", values="cn_b")[bins]

    if add_normal:
        cn_a = _add_normal_clone(cn_a)

        cn_b = _add_normal_clone(cn_b)

    cn_t = cn_a + cn_b

    print(f"Analysing using {cn_t.shape[0]} clones and {cn_t.shape[1]} bins")

    bins = list(cn_t.columns)

    clones = list(cn_t.index)

    data = {
        "num_clones": cn_t.shape[0],
        "num_bins": cn_t.shape[1],
        "cn_a": cn_a.to_numpy(),
        "cn_t": cn_t.to_numpy(),
        "a": a.to_numpy(),
        "d": d.to_numpy(),
        "rdr": rdr.to_numpy(),
    }

    return bins, clones, data


def _add_bin_name_col(df):
    df["bin_name"] = df["chrom"].astype(str) + ":" + df["start"].astype(str) + ":" + df["end"].astype(str)


def _add_normal_clone(df):
    clones = list(df.index)
    clones.append("normal")
    bins = df.columns
    vals = df.values
    vals = np.vstack([vals, np.ones(df.shape[1])])
    return pd.DataFrame(vals, index=clones, columns=bins)