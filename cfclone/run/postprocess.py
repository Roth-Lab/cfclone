from scipy.special import logsumexp as log_sum_exp
from skbio.tree import TreeNode

import arviz as az
import h5py
import xarray as xr
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as ss

from .json_serialisation import serialise_networkx_tree_to_json


def print_model_evidence(in_file):
    _, _, summary_df = _load_results(in_file)

    print(summary_df.iloc[-1]["stepping_stone"])


def preprocess_tree_to_extant_tips(clones, tree):
    index_set = set(clones)
    tip_set = set(tip.name for tip in tree.tips())
    tip_intersect = tip_set.intersection(index_set)
    return tree.shear(tip_intersect, prune=True, strict=False, inplace=False)


def assign_ancestral_node_names(clones, treefile):
    tree = TreeNode.read(treefile, convert_underscores=False)
    tree = preprocess_tree_to_extant_tips(clones, tree)
    template = "ancestral_{}"
    tree.assign_ids()

    ancestral_id_map = {}

    inner_node_idx = 0

    for node in tree.preorder():
        if node.is_tip():
            continue
        ancestral_id_map[node.id] = inner_node_idx
        inner_node_idx += 1

    for node in tree.postorder():

        if node.name is None:
            node.name = template.format(ancestral_id_map[node.id])

    return tree


def build_clone_df_and_tree(df, tree):
    df["clone_id"] = df["clone_id"].astype(str)
    df = df.set_index(["iteration", "chain"])
    clonal_grouped = df.groupby("clone_id")

    grp_dict = dict()
    for node in tree.postorder():
        if node.is_tip():
            grp_dict[node.name] = clonal_grouped.get_group(node.name)
            continue

        node_grp = grp_dict[node.children[0].name].copy()
        node_grp["clone_id"] = node.name

        for child in node.children[1:]:
            child_name = child.name
            child_grp = grp_dict[child_name]
            node_grp["rho"] += child_grp["rho"]

        grp_dict[node.name] = node_grp
    clone_df = pd.concat(grp_dict.values()).reset_index()
    return clone_df, tree


def scikit_tree_to_networkx_add_prev_and_hdi(tree, df_summary):
    nx_graph = nx.DiGraph()

    index_template = "rho[{}]"

    tip_label = "Clone {}\nClonal Prev: {}\nHDI lower: {} | HDI upper: {}"
    inner_label = "{}\nClonal Prev: {}\nHDI lower: {} | HDI upper: {}"

    for node in tree.traverse():

        node_name = node.name
        summary_row = df_summary.loc[index_template.format(node_name)]

        for child in node.children:
            child_name = child.name
            nx_graph.add_edge(node_name, child_name, length=child.length)

        hdi_low = summary_row["lower_hdi"]
        hdi_upper = summary_row["upper_hdi"]
        mean = summary_row["mean_prevalence"]

        if node.is_tip():
            label = tip_label.format(node_name, round(mean, 3), round(hdi_low, 3), round(hdi_upper, 3))
        else:
            label = inner_label.format(node_name, round(mean, 3), round(hdi_low, 3), round(hdi_upper, 3))

        nx_graph.add_node(node_name, label=label, prevalence_stats=summary_row)

    return nx_graph


def finalise_ancestral_rho_summary_df(df_summary, tree, sample_id=None):
    if sample_id is not None:
        df_summary["sample_id"] = sample_id
    df_summary = df_summary.reset_index(names=["parameters"])
    df_summary[["parameter_name", "clone_id"]] = df_summary["parameters"].str.split("[", expand=True)
    df_summary["clone_id"] = df_summary["clone_id"].str.removesuffix("]")
    df_summary.drop(columns=["parameters"], inplace=True)
    node_order_dict = {v.name: k for k, v in enumerate(tree.postorder())}
    df_summary = df_summary.set_index("clone_id")
    df_summary["clone_tree_order_idx"] = node_order_dict
    df_summary.reset_index(inplace=True)
    return df_summary


def compute_ancestral_prevalences(
    in_file,
    out_table,
    tree_json,
    clone_newick,
    normal=False,
    renormalise=True,
    hdi_prob=0.95,
    sample_id=None,
):
    data, samples_df, _ = _load_results(in_file)
    tree = assign_ancestral_node_names(data["clones"], clone_newick)

    rho_df = _build_rho_long_df(samples_df, keep_normal=normal, renormalise=renormalise)
    rho_df, tree = build_clone_df_and_tree(rho_df, tree)
    df_summary = build_arviz_rho_summary_df(rho_df, "rho", hdi_prob)
    hdi_col_name_map = _define_hdi_upper_and_lower_cols(df_summary, rename=True)
    _rename_arviz_summary_mean_median_cols(df_summary, "prevalence")

    nx_tree = scikit_tree_to_networkx_add_prev_and_hdi(tree, df_summary)

    if sample_id is not None:
        nx_tree.graph["sample_id"] = sample_id
    rev_hdi_col_map = {v: k for k, v in hdi_col_name_map.items()}
    nx_tree.graph["hdi_interval_width"] = hdi_prob
    nx_tree.graph["normal_clone_kept"] = normal
    nx_tree.graph["prevalences_renormalised"] = renormalise
    nx_tree.graph.update(rev_hdi_col_map)

    df_summary = finalise_ancestral_rho_summary_df(df_summary, tree, sample_id)

    df_summary.to_csv(out_table, sep="\t", index=False)
    serialise_networkx_tree_to_json(nx_tree, tree_json)


def write_parameter_summaries(
    in_file,
    out_file,
    hdi_prob=0.95,
):
    data, samples_df, _ = _load_results(in_file)

    mu_df, p_df = _build_mu_and_p_dfs(data, samples_df)

    print("mu and p dataframes built\n")

    baf = data["a"] / data["d"]

    mu_residual = mu_df.rsub(data["rdr"])

    p_residual = p_df.rsub(baf)

    print("mu and p residuals computed\n")

    rdr_outlier_df = _compute_rdr_outlier_prob(p_df, samples_df, data)

    baf_outlier_df = _compute_baf_outlier_prob(mu_df, samples_df, data)

    print("RDR and BAF outlier probs computed\n")

    iter_chain_df = samples_df[["iteration", "chain"]]

    mu_summary = _process_param_table(
        mu_df,
        iter_chain_df,
        hdi_prob,
        "mu",
        "mu",
    )

    p_summary = _process_param_table(
        p_df,
        iter_chain_df,
        hdi_prob,
        "p",
        "p",
    )

    mu_residual_summary = _process_param_table(
        mu_residual,
        iter_chain_df,
        hdi_prob,
        "mu",
        "mu_residual",
    )

    p_residual_summary = _process_param_table(
        p_residual,
        iter_chain_df,
        hdi_prob,
        "p",
        "p_residual",
    )

    baf_outlier_summary = _process_param_table(
        baf_outlier_df,
        iter_chain_df,
        hdi_prob,
        "baf_outlier",
        "baf_outlier_prob",
    )

    rdr_outlier_summary = _process_param_table(
        rdr_outlier_df,
        iter_chain_df,
        hdi_prob,
        "rdr_outlier",
        "rdr_outlier_prob",
    )

    result_df = mu_summary.join(
        [
            mu_residual_summary,
            p_summary,
            p_residual_summary,
            baf_outlier_summary,
            rdr_outlier_summary,
        ]
    )
    result_df["data_rdr"] = data["rdr"]

    result_df["data_baf"] = baf

    _add_bin_cols_to_summary_df(data["bins"], result_df)

    result_df.to_csv(out_file, sep="\t")


def _process_param_table(param_df, iter_chain_df, hdi_prob, param_name, col_prefix):
    param_df = iter_chain_df.join(param_df)

    mu_summary = _build_parameter_summary_df(param_df, hdi_prob, param_name, col_prefix)

    return mu_summary


def _compute_baf_outlier_prob(p_df, samples_df, data):
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

    result = np.exp(result)

    num_bins = p.shape[1]

    return pd.DataFrame(result, columns=["baf_outlier.{}".format(i) for i in range(num_bins)])


def _compute_rdr_outlier_prob(mu_df, samples_df, data):
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

    result = np.exp(result)

    num_bins = mu.shape[1]

    return pd.DataFrame(result, columns=["rdr_outlier.{}".format(i) for i in range(num_bins)])


def _add_bin_cols_to_summary_df(bins, df):
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

    prob_dom.to_csv(out_file, index=False, sep="\t")


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
        "lower_hdi": hdi[0],
        "upper_hdi": hdi[1],
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

    # TODO: handle cases where normal was dropped while being the only clone

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


def build_arviz_rho_summary_df(df, varname, hdi_prob, drop_sd_col=True):
    stats_funcs = {"median": np.median}
    df = df.rename(columns={"iteration": "draw"})
    df = df.set_index(["chain", "draw", "clone_id"])
    xdata = xr.Dataset.from_dataframe(df)
    az_dataset = az.InferenceData(posterior=xdata)
    df_summary = az.summary(
        az_dataset,
        var_names=[varname],
        kind="stats",
        hdi_prob=hdi_prob,
        round_to="none",
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
        hdi_col_name_map = {hdi_col_names[0]: "lower_hdi", hdi_col_names[1]: "upper_hdi"}
    else:
        hdi_col_name_map = {hdi_col_names[1]: "lower_hdi", hdi_col_names[0]: "upper_hdi"}
    if rename:
        df_summary.rename(columns=hdi_col_name_map, inplace=True)
    else:
        for hdi_col, new_name in hdi_col_name_map.items():
            df_summary[new_name] = df_summary[hdi_col]
    return hdi_col_name_map
