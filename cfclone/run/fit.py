import importlib.resources
import os

import numpy as np
import pandas as pd
import random

import cfclone.stan


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

    jl = setup_julia_module()

    reference, target = setup_model(jl, data)

    pt = run_inference(jl, reference, target, num_chains=num_chains, num_chains_vi=num_chains_vi, num_rounds=num_rounds)

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
    def add_bin_name_col(df):
        df["bin_name"] = df["chrom"].astype(str) + ":" + df["start"].astype(str) + ":" + df["end"].astype(str)

    def add_normal_clone(df):
        clones = list(df.index)
        clones.append("normal")
        bins = df.columns
        vals = df.values
        vals = np.row_stack([vals, np.ones(df.shape[1])])
        return pd.DataFrame(vals, index=clones, columns=bins)

    df = pd.read_csv(in_file, sep="\t")

    add_bin_name_col(df)

    clone_df = pd.read_csv(clone_cnv_file, sep="\t")

    add_bin_name_col(clone_df)

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
        cn_a = add_normal_clone(cn_a)

        cn_b = add_normal_clone(cn_b)

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


def setup_julia_module(num_threads=1):
    os.environ["JULIA_NUM_THREADS"] = f"{num_threads}"

    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

    import juliacall

    jl = juliacall.newmodule("cfClone")

    jl.seval("using ADTypes, Bijectors, BridgeStan, Distributions, DynamicPPL, JSON, Pigeons, ReverseDiff")

    jl.seval(
        """
    struct CfCloneDescription
    end
    """
    )

    return jl


def setup_model(jl, data, use_outlier=True):
    stan_dir = importlib.resources.files(cfclone.stan)

    if use_outlier:
        stan_file = stan_dir.joinpath("cfclone_outlier.stan")

        build_reference = jl.seval(
            """
        function build_reference(n_clones)
            result = []
            for _ in 1:(n_clones - 1)
                push!(result, Uniform(0, 1)) # rho
            end
            append!(result,
                    [Distributions.Gamma(1, 1),     # alpha
                    Distributions.Beta(1, 100),     # non_binomiality
                    Distributions.Gamma(1, 100),    # sigma
                    Distributions.Gamma(1, 1),      # sigma_outlier 
                    Distributions.Gamma(1, 100),    # outlier_rate_rdr
                    Distributions.Gamma(1, 100)]    # outlier_rate_baf
            )
            return DistributionLogPotential(product_distribution(transformed.(result)))
        end
        """
        )

    else:
        stan_file = stan_dir.joinpath("cfclone.stan")

        build_reference = jl.seval(
            """
        function build_reference(n_clones)
            result = []
            for _ in 1:(n_clones - 1)
                push!(result, Uniform(0, 1)) # rho
            end
            append!(result,
                    [Distributions.Gamma(1, 1),     # alpha
                    Distributions.Beta(1, 100),     # non_binomiality
                    Distributions.Gamma(1, 100)]    # sigma
            )
            return DistributionLogPotential(product_distribution(transformed.(result)))
        end
        """
        )

    build_target = jl.seval(
        """
    function build_target(data, stan_model_file)
        description = CfCloneDescription() 
        return StanLogPotential(stan_model_file, JSON.json(data), description)
    end
    """
    )

    reference = build_reference(data["num_clones"])

    target = build_target(data, str(stan_file))

    return reference, target


def run_inference(jl, reference, target, num_chains=12, num_chains_vi=5, num_rounds=10):
    infer = jl.seval(
        """
    function infer(reference, target, n_chains=12, n_chains_variational=5, n_rounds=10)
        result = pigeons(
            ;
            target,
            record=[traces, Pigeons.round_trip, Pigeons.timing_extrema, Pigeons.energy_ac1],
            explorer=AutoMALA(),
            n_chains,
            reference,
            n_rounds,
            n_chains_variational=5,
            variational=GaussianReference(first_tuning_round=5)
        )
        return result
    end
    """
    )

    return infer(
        reference,
        target,
        num_chains,
        num_chains_vi,
        num_rounds,
    )
