import os
import random

import h5py
import numpy as np
import pandas as pd

from cfclone.inference import run_inference
from cfclone.julia import setup_julia_module
from cfclone.models import get_model
from cfclone.run.initialise import set_env_variables


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
    only_normal=False,
    outlier=False,
    use_clone=(),
):
    bins, clones, data = load_data(
        clone_cnv_file,
        in_file,
        add_normal=add_normal,
        num_bins=num_bins,
        only_normal=only_normal,
        use_clone=use_clone,
    )

    print(clones)

    set_env_variables()

    os.environ["PYTHON_JULIACALL_THREADS"] = f"{num_threads}"

    jl = setup_julia_module()

    print("\nUsing {} threads\n".format(jl.Threads.nthreads()))

    model = get_model(jl, data, use_outlier=outlier)

    pt = run_inference(
        jl,
        model,
        num_chains=num_chains,
        num_chains_vi=num_chains_vi,
        num_rounds=num_rounds,
    )

    samples_df = build_samples_df(clones, jl, pt)

    summary_df = build_summary_df(jl, pt)

    with h5py.File(out_file, "w") as fh:
        fh.create_dataset("/data/bins", data=bins, dtype=h5py.string_dtype(encoding="utf-8"))

        fh.create_dataset("/data/clones", data=[str(x) for x in clones], dtype=h5py.string_dtype(encoding="utf-8"))

        dset = fh.create_dataset("/data/cn_a", data=data["cn_a"], dtype=np.int32)

        dset = fh.create_dataset("/data/cn_t", data=data["cn_t"], dtype=np.int32)

        dset = fh.create_dataset("/data/a", data=data["a"], dtype=np.int32)

        dset = fh.create_dataset("/data/d", data=data["d"], dtype=np.int32)

        dset = fh.create_dataset("/data/rdr", data=data["rdr"], dtype=np.float64)

        dset = fh.create_dataset("/results/samples", data=samples_df.to_numpy(), dtype=np.float64)
        dset.attrs["columns"] = [str(x) for x in samples_df.columns]

        dset = fh.create_dataset("/results/summary", data=summary_df.to_numpy(), dtype=np.float64)
        dset.attrs["columns"] = [str(x) for x in summary_df.columns]


def build_summary_df(jl, pt):
    summary_df = jl.PythonCall.pytable(pt.shared.reports.summary)
    summary_df = summary_df.drop("last_round_max_allocation", axis=1)
    summary_df = summary_df.astype(np.float64)
    return summary_df


def build_samples_df(clones, jl, pt):
    chains = jl.Chains(pt)
    samples_df = jl.PythonCall.pytable(chains)
    clone_dict = {i + 1: clone for i, clone in enumerate(clones)}
    rho_map = {col: "rho_{}".format(clone_dict[int(col[4:])]) for col in samples_df.columns if col.startswith("rho.")}
    samples_df.rename(columns=rho_map, inplace=True)
    return samples_df


def load_data(clone_cnv_file, in_file, add_normal=True, num_bins=None, only_normal=False, use_clone=()):
    df = pd.read_csv(in_file, sep="\t")

    _add_bin_name_col(df)

    clone_df = pd.read_csv(clone_cnv_file, converters={"clone": str}, sep="\t")

    _add_bin_name_col(clone_df)

    # Ensure the same set of bins is used and data is aligned
    bins = pd.merge(
        df[["bin_name"]].drop_duplicates(), clone_df[["bin_name"]].drop_duplicates(), on="bin_name", how="inner"
    )["bin_name"]

    if num_bins is not None and num_bins <= len(bins):
        bins = random.sample(list(bins), num_bins)

    a = df.set_index("bin_name").loc[bins, "a"]

    b = df.set_index("bin_name").loc[bins, "b"]

    d = a + b

    rdr = df.set_index("bin_name").loc[bins, "rdr"]

    cn_a = clone_df.pivot(index="clone", columns="bin_name", values="cn_a")[bins]

    cn_b = clone_df.pivot(index="clone", columns="bin_name", values="cn_b")[bins]

    if len(use_clone) > 0:
        # print(use_clone)
        clone_list = [x for x in use_clone]
        cn_a = cn_a.loc[clone_list]

        cn_b = cn_b.loc[clone_list]

    if add_normal:
        cn_a = _add_normal_clone(cn_a)

        cn_b = _add_normal_clone(cn_b)

    if only_normal:
        cn_a = cn_a.loc[["normal"]]

        cn_b = cn_b.loc[["normal"]]

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
    vals = df.to_numpy()
    vals = np.vstack([vals, np.ones(df.shape[1])])
    return pd.DataFrame(vals, index=clones, columns=bins)
