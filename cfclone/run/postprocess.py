import arviz as az
import h5py
import xarray as xr
import numpy as np
import pandas as pd
from scipy.special import logsumexp as log_sum_exp
import scipy.stats as ss


def print_model_evidence(in_file):
    _, _, summary_df = _load_results(in_file)

    print(summary_df.iloc[-1]["stepping_stone"])


def write_parameter_summaries(
    in_file,
    out_file,
    hdi_prob=0.95,
):
    data, samples_df, _ = _load_results(in_file)
    mu_df, p_df = _build_mu_and_p_dfs(data, samples_df)

    print("mu and p dataframes built")

    baf = data["a"] / data["d"]
    # rdr = data["rdr"]

    mu_residual = mu_df.rsub(data["rdr"])
    p_residual = p_df.rsub(baf)

    print("mu and p residuals computed\n")

    rdr_outlier_df = _compute_rdr_outlier_prob(p_df, samples_df, data)
    baf_outlier_df = _compute_baf_outlier_prob(mu_df, samples_df, data)

    print("RDR and BAF outlier probs computed\n")

    iter_chain_df = samples_df[["iteration", "chain"]]

    mu_summary = _process_param_table(mu_df, iter_chain_df, hdi_prob, "mu", "mu")
    p_summary = _process_param_table(p_df, iter_chain_df, hdi_prob, "p", "p")
    mu_residual_summary = _process_param_table(mu_residual, iter_chain_df, hdi_prob, "mu", "mu_residual")
    p_residual_summary = _process_param_table(p_residual, iter_chain_df, hdi_prob, "p", "p_residual")
    baf_outlier_summary = _process_param_table(baf_outlier_df, iter_chain_df, hdi_prob, "baf_outlier", "baf_outlier_prob")
    rdr_outlier_summary = _process_param_table(rdr_outlier_df, iter_chain_df, hdi_prob, "rdr_outlier", "rdr_outlier_prob")

    # p_df = iter_chain_df.join(p_df)
    # mu_residual = iter_chain_df.join(mu_residual)
    # p_residual = iter_chain_df.join(p_residual)
    # rdr_outlier_df = iter_chain_df.join(rdr_outlier_df)
    # baf_outlier_df = iter_chain_df.join(baf_outlier_df)

    # p_summary = _build_parameter_summary_df(p_df, hdi_prob, "p", "p")
    # mu_residual_summary = _build_parameter_summary_df(mu_residual, hdi_prob, "mu", "mu_residual")
    # p_residual_summary = _build_parameter_summary_df(p_residual, hdi_prob, "p", "p_residual")
    # baf_outlier_summary = _build_parameter_summary_df(baf_outlier_df, hdi_prob, "baf_outlier", "baf_outlier_prob")
    # rdr_outlier_summary = _build_parameter_summary_df(rdr_outlier_df, hdi_prob, "rdr_outlier", "rdr_outlier_prob")

    result_df = mu_summary.join([mu_residual_summary, p_summary, p_residual_summary, baf_outlier_summary, rdr_outlier_summary])
    add_bin_cols_to_summary_df(data["bins"], result_df)

    result_df.to_csv(out_file, sep="\t")


def _process_param_table(param_df, iter_chain_df, hdi_prob, param_name, col_prefix):
    param_df = iter_chain_df.join(param_df)
    mu_summary = _build_parameter_summary_df(param_df, hdi_prob, param_name, col_prefix)
    return mu_summary


def _compute_rdr_outlier_prob(p_df, samples_df, data):
    p = p_df.to_numpy()
    w = samples_df["outlier_rate_rdr"].to_numpy()
    n = samples_df["non_binomiality"].to_numpy()[..., np.newaxis]
    data_a = data["a"]
    data_d = data["d"]

    a_tmp = p / n
    b_tmp = (1 - p) / n

    log_w = np.log(w)[..., np.newaxis]
    log_w_minus_1 = np.log1p(-w)[..., np.newaxis]

    beta_binom_ones = ss.betabinom.pmf(data_a, data_d, 1, 1)

    beta_binom_a_b = ss.betabinom.pmf(data_a, data_d, a_tmp, b_tmp)

    stacked = np.stack([log_w + beta_binom_ones, log_w_minus_1 + beta_binom_a_b])

    result = stacked[0] - log_sum_exp(stacked, axis=0)

    num_bins = p.shape[1]
    rdr_outlier_df = pd.DataFrame(result, columns=["rdr_outlier.{}".format(i) for i in range(num_bins)])
    return rdr_outlier_df


def _compute_baf_outlier_prob(mu_df, samples_df, data):
    mu = mu_df.to_numpy()
    s = samples_df["sigma"].to_numpy()[..., np.newaxis]
    s_outlier = samples_df["sigma_outlier"].to_numpy()[..., np.newaxis]
    w = samples_df["outlier_rate_rdr"].to_numpy()

    data_rdr = data["rdr"]

    log_w = np.log(w)[..., np.newaxis]
    log_w_minus_1 = np.log1p(-w)[..., np.newaxis]

    s_outlier_pdf = ss.t.logpdf(data_rdr, 25, 0, s_outlier)

    mu_s_pdf = ss.t.logpdf(data_rdr, 25, mu, s)

    stacked = np.stack([log_w + s_outlier_pdf, log_w_minus_1 + mu_s_pdf])

    result = stacked[0] - log_sum_exp(stacked, axis=0)

    num_bins = mu.shape[1]
    baf_outlier_df = pd.DataFrame(result, columns=["baf_outlier.{}".format(i) for i in range(num_bins)])
    return baf_outlier_df


def add_bin_cols_to_summary_df(bins, df):
    df["bin_name"] = bins
    df[["chrom", "start", "end"]] = df["bin_name"].str.split(":", expand=True)
    df.drop(columns="bin_name", inplace=True)


def _build_parameter_summary_df(df, hdi_prob, param_name, col_prefix):
    out_df = _build_arviz_summary_df_long(df, param_name, hdi_prob)
    print("{} summary dataframe built".format(col_prefix))
    _define_hdi_upper_and_lower_cols(out_df, rename=True)
    out_df.index = out_df.index.str.removeprefix("{}.".format(param_name))
    out_df.index.name = "bin_idx"
    out_df.index = pd.to_numeric(out_df.index, downcast="integer")
    out_df.sort_index(axis=0, inplace=True)
    out_df = out_df.add_prefix("{}_".format(col_prefix), axis=1)
    print("{} summary col processing complete\n".format(col_prefix))
    return out_df


def write_dominance_prob(in_file, out_file, normal=False):
    data, samples_df, _ = _load_results(in_file)

    renormalise = _should_renormalise(normal)

    rho_df, rho_cols = _build_rho_wide_df(samples_df, keep_normal=normal, renormalise=renormalise)

    rho_df["max_rho"] = rho_df[rho_cols].max(axis=1)

    rho_df = pd.wide_to_long(
        rho_df,
        stubnames="rho",
        sep="_",
        i=["iteration", "chain"],
        j="clone_id",
        suffix="(\\d+|normal)",
    )

    rho_df.reset_index(inplace=True)

    rho_df["is_max"] = rho_df["rho"] == rho_df["max_rho"]

    prob_dom = rho_df.groupby("clone_id", sort=False)["is_max"].mean()

    prob_dom = prob_dom.reset_index()

    prob_dom.rename(columns={"is_max": "dominance_prob"}, inplace=True)

    prob_dom.index.name = "clone"

    prob_dom.to_csv(out_file, sep="\t")


def write_pairwise_ranks(in_file, out_file, normal=False):
    data, samples_df, _ = _load_results(in_file)

    rho_df = _build_rho_long_df(samples_df, normal, renormalise=False)

    post_df = _build_dominance_df(rho_df)

    post_df.to_csv(out_file, sep="\t")


def write_prevalence_samples(in_file, out_file):
    _, samples_df, _ = _load_results(in_file)

    out_df, rho_cols = _build_rho_wide_df(samples_df, keep_normal=True, renormalise=False)

    out_df.columns = out_df.columns.str.replace("rho", "clone")

    out_df.to_csv(out_file, index=False, sep="\t")


def write_prevalence_stats(in_file, out_file, hdi_prob=0.95, normal=False, renormalise=True):
    data, samples_df, _ = _load_results(in_file)

    rho_df, rho_cols = _build_rho_wide_df(samples_df, keep_normal=normal, renormalise=renormalise)
    out_df = _build_arviz_summary_df_long(rho_df, "rho", hdi_prob)
    _define_hdi_upper_and_lower_cols(out_df, rename=True)
    _rename_arviz_summary_mean_median_cols(out_df, "prevalence")

    out_df.index = out_df.index.str.removeprefix("rho_")
    out_df.index.name = "clone_id"

    out_df.to_csv(out_file, sep="\t")


def _rename_arviz_summary_mean_median_cols(df, suffix_to_add):
    suffix_to_add = "_{}".format(suffix_to_add)
    colmap = {"mean": "mean{}".format(suffix_to_add), "median": "median{}".format(suffix_to_add)}
    df.rename(columns=colmap, inplace=True)


def write_samples(in_file, out_file, compute_generated_quantities=False):
    data, samples_df, _ = _load_results(in_file)

    if compute_generated_quantities:
        samples_df = _create_mu_and_p_cols(data, samples_df)

    samples_df.to_csv(out_file, index=False, sep="\t")


def write_summary(in_file, out_file):
    _, _, summary_df = _load_results(in_file)

    summary_df.to_csv(out_file, index=False, sep="\t")


def write_tumour_content(in_file, out_file, hdi_prob=0.95):
    data, samples_df, _ = _load_results(in_file)

    rho_df, _ = _build_rho_wide_df(samples_df, keep_normal=True, renormalise=False)

    rho_df["tumour_content"] = 1 - rho_df[["rho_normal"]]

    hdi = az.hdi(rho_df["tumour_content"].to_numpy(), hdi_prob=hdi_prob)

    out_record = {
        "mean": rho_df["tumour_content"].mean(),
        "median": rho_df["tumour_content"].median(),
        "lower_ci": hdi[0],
        "upper_ci": hdi[1],
    }

    out_df = pd.DataFrame([out_record])

    out_df.to_csv(out_file, index=False, sep="\t")


def _load_results(file_name):
    data = {}

    with h5py.File(file_name) as fh:
        samples_df = _load_df(fh, "/results/samples", downcast=True)

        summary_df = _load_df(fh, "/results/summary")

        data["bins"] = [x.decode() for x in fh["/data/bins"][()]]

        data["clones"] = [x.decode() for x in fh["/data/clones"][()]]

        data["a"] = fh["/data/a"][()]

        data["d"] = fh["/data/d"][()]

        data["rdr"] = fh["/data/rdr"][()]

        data["cn_a"] = fh["/data/cn_a"][()]

        data["cn_t"] = fh["/data/cn_t"][()]

    return data, samples_df, summary_df


def _load_df(fh, key, downcast=False):
    vals = fh[key][()]

    cols = fh[key].attrs["columns"]

    df = pd.DataFrame(vals, columns=cols)

    if downcast:
        df[["iteration", "chain"]] = df[["iteration", "chain"]].apply(pd.to_numeric, downcast="integer")

    return df


def _build_dominance_df(df):
    df = df[["iteration", "chain", "clone_id", "rho"]]

    df = df.set_index(["iteration", "chain"])

    clonal_grouped = df.groupby("clone_id", sort=False)

    new_df = []

    for i_name, i_group in clonal_grouped:
        i_records = {"clone_id": i_name}

        total_len = len(i_group)

        for j_name, j_group in clonal_grouped:
            diff_df = i_group["rho"] > j_group["rho"]

            diff_val = diff_df.sum() / total_len

            i_records[j_name] = diff_val

        new_df.append(i_records)

    new_df = pd.DataFrame(new_df)

    new_df = new_df.set_index("clone_id")

    return new_df


def _build_rho_long_df(samples_df, keep_normal, renormalise):
    df, rho_cols = _build_rho_wide_df(samples_df, keep_normal, renormalise)

    df = pd.wide_to_long(
        df,
        stubnames="rho",
        sep="_",
        i=["iteration", "chain"],
        j="clone_id",
        suffix="(\\d+|normal)",
    )

    df.reset_index(inplace=True)

    return df


def _build_rho_wide_df(samples_df, keep_normal, renormalise, keep_cols=None):
    if not keep_normal:
        df = samples_df.drop(columns="rho_normal", errors="ignore")

    else:
        df = samples_df

    rho_cols = [col for col in df.columns if col.startswith("rho")]

    if renormalise and not keep_normal:
        df["rho_sum"] = df[rho_cols].sum(axis=1)

        df[rho_cols] = df[rho_cols].div(df["rho_sum"], axis=0)

        df = df.drop(columns="rho_sum")

    cols_to_keep = ["iteration", "chain"]

    cols_to_keep.extend(rho_cols)

    if keep_cols is not None:
        if isinstance(keep_cols, str):
            cols_to_keep.append(keep_cols)
        else:
            cols_to_keep.extend(keep_cols)

    df = df[cols_to_keep].copy()

    return df, rho_cols


def _create_mu_and_p_cols(data, samples_df):
    mu_df, p_df = _build_mu_and_p_dfs(data, samples_df)

    samples_df = samples_df.join([mu_df, p_df])

    return samples_df


def _build_mu_and_p_dfs(data, samples_df):
    cn_t = data["cn_t"]
    cn_a = data["cn_a"]
    rho_df, rho_cols = _build_rho_wide_df(samples_df, keep_normal=True, renormalise=False, keep_cols="alpha")

    rho = rho_df[rho_cols].to_numpy()
    mean_clone_cn = np.mean(cn_t, axis=1)
    alpha = rho_df["alpha"].to_numpy()
    alpha = alpha[..., np.newaxis]

    rho_mat_cn_t = rho @ cn_t

    mu = rho_mat_cn_t * alpha
    mu /= np.matvec(rho, mean_clone_cn)[..., np.newaxis]

    p = rho @ cn_a
    p /= rho_mat_cn_t

    num_bins = cn_t.shape[1]

    mu_df = pd.DataFrame(mu, columns=["mu.{}".format(i) for i in range(num_bins)])
    p_df = pd.DataFrame(p, columns=["p.{}".format(i) for i in range(num_bins)])
    return mu_df, p_df


def _should_renormalise(normal):
    if normal:
        renormalise = False

    else:
        renormalise = True

    return renormalise


def _build_arviz_summary_df_long(df, varname, hdi_prob, drop_sd_col=True):
    stats_funcs = {"median": np.median}
    df = df.rename(columns={"iteration": "draw"})
    df = df.set_index(["chain", "draw"])
    xdata = xr.Dataset.from_dataframe(df)
    az_dataset = az.InferenceData(posterior=xdata)
    df_summary = az.summary(
        az_dataset,
        var_names=[varname],
        kind="stats",
        hdi_prob=hdi_prob,
        round_to="none",
        filter_vars="like",
        extend=True,
        stat_funcs=stats_funcs,
    )
    if drop_sd_col:
        df_summary.drop(columns="sd", inplace=True)
    return df_summary


def _define_hdi_upper_and_lower_cols(df_summary, rename=True):
    hdi_cols = {col: float(col[4:-1]) for col in df_summary.columns if col.startswith("hdi")}
    hdi_col_names = list(hdi_cols.keys())
    if hdi_cols[hdi_col_names[0]] < hdi_cols[hdi_col_names[1]]:
        hdi_col_name_map = {hdi_col_names[0]: "lower_ci", hdi_col_names[1]: "upper_ci"}
    else:
        hdi_col_name_map = {hdi_col_names[1]: "lower_ci", hdi_col_names[0]: "upper_ci"}
    if rename:
        df_summary.rename(columns=hdi_col_name_map, inplace=True)
    else:
        for hdi_col, new_name in hdi_col_name_map.items():
            df_summary[new_name] = df_summary[hdi_col]
