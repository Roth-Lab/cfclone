import enum
import os
import pathlib
import random

import h5py
import numpy as np
import pandas as pd

from cfclone.julia import run_inference, setup_julia_module


class SexType(enum.Enum):
    female = enum.auto()
    male = enum.auto()


def fit(
    clone_cnv_file,
    in_file,
    out_file,
    add_normal=True,
    exec_dir=None,
    laplace_exec_dir=None,
    num_bins=None,
    num_chains=12,
    num_rounds=10,
    num_threads=1,
    pi_normal=10,
    pi_tumour=0.1,
    only_normal=False,
    outlier=False,
    seed=None,
    sex=SexType.female,
    use_clone=(),
):
    priors = {
        "pi_normal": pi_normal,
        "pi_tumour": pi_tumour,
    }

    rng = np.random.default_rng(seed)

    bins, clones, data = load_data(
        clone_cnv_file,
        in_file,
        priors,
        rng,
        add_normal=add_normal,
        num_bins=num_bins,
        only_normal=only_normal,
        sex=sex,
        use_clone=use_clone,
    )

    print(clones)

    jl = setup_julia_module(num_threads=num_threads)

    pt_seed = rng.integers(int(1e8))

    pt, ls = run_inference(
        jl,
        data,
        pt_seed,
        exec_dir=exec_dir,
        laplace_exec_dir=laplace_exec_dir,
        num_chains=num_chains,
        num_rounds=num_rounds,
        num_threads=num_threads,
        use_outlier=outlier,
    )

    samples_df = build_samples_df(clones, jl, pt)
    
    laplace_samples_df = build_laplace_samples_df(clones, jl, ls)
    
    laplace_opt_trace_df = build_laplace_trace_df(clones, jl, ls)

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
        
        dset = fh.create_dataset("/results/laplace_samples", data=laplace_samples_df.to_numpy(), dtype=np.float64)
        dset.attrs["columns"] = [str(x) for x in laplace_samples_df.columns]
        
        dset = fh.create_dataset("/results/laplace_opt_trace", data=laplace_opt_trace_df.to_numpy(), dtype=np.float64)
        dset.attrs["columns"] = [str(x) for x in laplace_opt_trace_df.columns]


    if exec_dir is not None:
        build_report = jl.seval(
            """
        function build_report(exec_folder, pt; interval_probability=0.95)
            report(pt; exec_folder,  interval_probability, max_dim=50, view=false)
        end
        """
        )

        build_report(exec_dir, pt)

        exec_dir_path = pathlib.Path(exec_dir)

        raw_report_path = exec_dir_path.joinpath("build", "index.html")

        report_path = exec_dir_path.joinpath("report.html")

        report_path.unlink(missing_ok=True)

        report_path.symlink_to(raw_report_path)


def build_summary_df(jl, pt):
    summary_df = jl.PythonCall.pytable(pt.shared.reports.summary)
    summary_df = summary_df.drop(["global_barrier_variational", "last_round_max_allocation"], axis=1)
    summary_df = summary_df.astype(np.float64)
    return summary_df


def build_samples_df(clones, jl, pt):
    chains = jl.Chains(pt)
    samples_df = jl.PythonCall.pytable(chains)
    clone_dict = {i + 1: clone for i, clone in enumerate(clones)}
    rho_map = {col: "rho_{}".format(clone_dict[int(col[4:])]) for col in samples_df.columns if col.startswith("rho.")}
    samples_df.rename(columns=rho_map, inplace=True)
    return samples_df

def build_laplace_samples_df(clones, jl, ls):
    samples_df = jl.PythonCall.pytable(ls.chains)
    clone_dict = {i + 1: clone for i, clone in enumerate(clones)}
    rho_map = {col: "rho_{}".format(clone_dict[int(col[4:])]) for col in samples_df.columns if col.startswith("rho.")}
    samples_df.rename(columns=rho_map, inplace=True)
    return samples_df


def build_laplace_trace_df(clones, jl, ls):
    log_density = np.asarray(ls.approximation.optimization_trace.values)
    step_sizes = np.asarray(ls.approximation.optimization_trace.step_sizes)
    # points = np.array([np.asarray(p) for p in ls.approximation.optimization_trace.points]) # POINTS ARE PARAMETERS IN UNCONSTRAINED SPACE
    out_df = pd.DataFrame({'iteration': range(1, len(log_density) + 1), 'log_density': log_density, 'step_size': step_sizes})
    return out_df

def load_data(
    clone_cnv_file,
    in_file,
    priors,
    rng,
    add_normal=True,
    num_bins=None,
    only_normal=False,
    sex=SexType.female,
    use_clone=(),
):
    df = pd.read_csv(in_file, sep="\t")

    _add_bin_name_col(df)

    clone_df = pd.read_csv(clone_cnv_file, converters={"clone": str}, sep="\t")

    _add_bin_name_col(clone_df)

    # Ensure the same set of bins is used and data is aligned
    bins = pd.merge(
        df[["bin_name"]].drop_duplicates(), clone_df[["bin_name"]].drop_duplicates(), on="bin_name", how="inner"
    )["bin_name"]

    if num_bins is not None and num_bins <= len(bins):
        bins = rng.choice(list(bins), size=num_bins, replace=False)

    a = df.set_index("bin_name").loc[bins, "a"]

    b = df.set_index("bin_name").loc[bins, "b"]

    d = a + b

    rdr = df.set_index("bin_name").loc[bins, "rdr"]

    cn_a = clone_df.pivot(index="clone", columns="bin_name", values="cn_a")[bins]

    cn_b = clone_df.pivot(index="clone", columns="bin_name", values="cn_b")[bins]

    if len(use_clone) > 0:
        # print(use_clone)
        clone_list = list({x for x in use_clone})
        cn_a = cn_a.loc[clone_list]

        cn_b = cn_b.loc[clone_list]

    if add_normal:
        cn_a = _add_normal_clone(cn_a)

        cn_b = _add_normal_clone(cn_b)

    if sex == SexType.female:
        cn_a.loc["normal", [x for x in cn_a.columns if x.split(":")[0].replace("chr", "") == "Y"]] = 0

        cn_b.loc["normal", [x for x in cn_b.columns if x.split(":")[0].replace("chr", "") == "Y"]] = 0

    else:
        cn_a.loc["normal", [x for x in cn_a.columns if x.split(":")[0].replace("chr", "") in ["X", "Y"]]] = 1

        cn_b.loc["normal", [x for x in cn_b.columns if x.split(":")[0].replace("chr", "") in ["X", "Y"]]] = 0

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

    data["pi"] = priors["pi_tumour"] * np.ones(data["num_clones"])

    if add_normal:
        data["pi"][-1] = priors["pi_normal"]

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
