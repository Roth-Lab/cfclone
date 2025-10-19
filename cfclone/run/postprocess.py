import arviz
import h5py
import numpy as np
import pandas as pd


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

    prob_dom.to_csv(out_file, sep="\t")


# def write_dominance_prob(in_file, out_file, normal=False):
#     data, samples_df, _ = _load_results(in_file)
#
#     clones, rho = _load_rho(data, samples_df, normal=normal)
#
#     num_scans, num_clones = rho.shape
#
#     post_mat = np.zeros((num_clones, num_clones))
#
#     prob_dom = (rho == rho.max(axis=1)[:, np.newaxis]).mean(axis=0)
#
#     prob_dom = pd.Series(prob_dom, index=clones)
#
#     prob_dom.to_csv(out_file, sep="\t")


# def write_pairwise_ranks(in_file, out_file, normal=False):
#     data, samples_df, _ = _load_results(in_file)
#
#     clones, rho = _load_rho(data, samples_df, normal=normal)
#
#     num_scans, num_clones = rho.shape
#
#     post_mat = np.zeros((num_clones, num_clones))
#
#     for t in range(num_scans):
#         for i in range(num_clones):
#             for j in range(num_clones):
#                 post_mat[i, j] += rho[t, i] >= rho[t, j]
#
#     post_mat /= num_scans
#
#     post_df = pd.DataFrame(post_mat, columns=clones, index=clones)
#
#     post_df.to_csv(out_file, sep="\t")


def write_pairwise_ranks(in_file, out_file, normal=False):
    data, samples_df, _ = _load_results(in_file)

    renormalise = _should_renormalise(normal)

    rho_df = _build_rho_long_df(samples_df, normal, renormalise)

    post_df = _build_dominance_df(rho_df)

    post_df.to_csv(out_file, sep="\t")


def _should_renormalise(normal):
    if normal:
        renormalise = False
    else:
        renormalise = True
    return renormalise


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


def _build_rho_wide_df(samples_df, keep_normal, renormalise):
    if not keep_normal:
        df = samples_df.drop(columns="rho_normal", errors="ignore")
    else:
        df = samples_df
    rho_cols = [col for col in df.columns if col.startswith("rho")]
    if renormalise:
        df["rho_sum"] = df[rho_cols].sum(axis=1)
        df[rho_cols] = df[rho_cols].div(df["rho_sum"], axis=0)
        df = df.drop(columns="rho_sum")
    cols_to_keep = ["iteration", "chain"]
    cols_to_keep.extend(rho_cols)
    df = df[cols_to_keep].copy()
    return df, rho_cols


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


# def write_prevalence_samples(in_file, out_file):
#     _, samples_df, _ = _load_results(in_file)
#
#     out_df = samples_df[[x for x in samples_df.columns if "rho" in x]]
#
#     out_df = out_df.rename(columns=lambda x: x.replace("rho", "clone"))
#
#     out_df.to_csv(out_file, index=False, sep="\t")


def write_prevalence_samples(in_file, out_file):
    _, samples_df, _ = _load_results(in_file)

    out_df, rho_cols = _build_rho_wide_df(samples_df, keep_normal=True, renormalise=False)

    out_df = out_df.rename(columns=lambda x: x.replace("rho", "clone"))

    out_df.to_csv(out_file, index=False, sep="\t")


# def write_prevalence_stats(in_file, out_file, hdi_prob=0.95, normal=False, renormalise=True):
#     data, samples_df, _ = _load_results(in_file)
#
#     clones, rho = _load_rho(data, samples_df, normal=normal, renormalise=renormalise)
#
#     hdi = arviz.hdi(rho.reshape((1, rho.shape[0], rho.shape[1])), hdi_prob=hdi_prob)
#
#     out_df = np.column_stack([np.mean(rho, axis=0), np.median(rho, axis=0), hdi])
#
#     out_df = pd.DataFrame(
#         out_df, columns=["mean_prevalence", "median_prevalence", "lower_ci", "upper_ci"], index=clones
#     )
#
#     out_df.index.name = "clone"
#
#     out_df.to_csv(out_file, sep="\t")


def write_prevalence_stats(in_file, out_file, hdi_prob=0.95, normal=False, renormalise=True):
    data, samples_df, _ = _load_results(in_file)

    rho_df, rho_cols = _build_rho_wide_df(samples_df, keep_normal=normal, renormalise=renormalise)

    rho = rho_df[rho_cols].to_numpy()

    hdi = arviz.hdi(rho.reshape((1, rho.shape[0], rho.shape[1])), hdi_prob=hdi_prob)

    out_record = {
        "mean_prevalence": rho_df[rho_cols].mean(axis=0),
        "median_prevalence": rho_df[rho_cols].median(axis=0),
        "lower_ci": hdi[:, 0],
        "upper_ci": hdi[:, 1],
    }

    out_df = pd.DataFrame.from_records(
        out_record,
        columns=["mean_prevalence", "median_prevalence", "lower_ci", "upper_ci"],
    )
    out_df.index = out_df.index.str.removeprefix("rho_")
    out_df.index.name = "clone_id"

    out_df.to_csv(out_file, sep="\t")


def write_samples(in_file, out_file, compute_generated_quantities=True):
    data, samples_df, _ = _load_results(in_file)

    if compute_generated_quantities:
        samples_df = _create_mu_and_p_cols(data, samples_df)

    samples_df.to_csv(out_file, index=False, sep="\t")


def _create_mu_and_p_cols(data, samples_df):
    cn_t = data["cn_t"]
    cn_a = data["cn_a"]
    rho_df, rho_cols = _build_rho_wide_df(samples_df, keep_normal=True, renormalise=False)
    rho = rho_df[rho_cols].to_numpy()
    rho = rho[..., np.newaxis]
    num_bins = cn_t.shape[1]
    num_sample_draws = rho.shape[0]
    mean_clone_cn = np.mean(cn_t, axis=1)
    mu_arr = np.empty((num_sample_draws, num_bins))
    p_arr = np.empty((num_sample_draws, num_bins))
    for i in range(num_sample_draws):
        rho_draw = rho[i]
        cn_t_by_rho = cn_t * rho_draw
        (cn_t_by_rho / np.dot(mean_clone_cn, rho_draw)).sum(axis=0, out=mu_arr[i])
        ((cn_a * rho_draw) / cn_t_by_rho).sum(axis=0, out=p_arr[i])
    mu_df = pd.DataFrame(mu_arr, columns=["mu.{}".format(i) for i in range(1, num_bins + 1)])
    p_df = pd.DataFrame(p_arr, columns=["p.{}".format(i) for i in range(1, num_bins + 1)])
    samples_df = samples_df.join([mu_df, p_df])
    return samples_df


# def write_tumour_content(in_file, out_file, hdi_prob=0.95):
#     data, samples_df, _ = _load_results(in_file)
#
#     clones, rho = _load_rho(data, samples_df, normal=True, renormalise=False)
#
#     rho = pd.DataFrame(rho, columns=clones)
#
#     rho = rho[["normal"]]
#
#     tc = (1 - rho[["normal"]]).values
#
#     hdi = arviz.hdi(tc.reshape((1, tc.shape[0], tc.shape[1])), hdi_prob=hdi_prob)
#
#     out_df = np.column_stack([np.mean(tc, axis=0), np.median(tc, axis=0), hdi])
#
#     out_df = pd.DataFrame(out_df, columns=["mean", "median", "lower_ci", "upper_ci"])
#
#     out_df.to_csv(out_file, index=False, sep="\t")


def write_tumour_content(in_file, out_file, hdi_prob=0.95):
    data, samples_df, _ = _load_results(in_file)

    rho_df, _ = _build_rho_wide_df(samples_df, keep_normal=True, renormalise=False)

    rho_df["tumour_content"] = 1 - rho_df[["rho_normal"]]

    hdi = arviz.hdi(rho_df["tumour_content"].to_numpy(), hdi_prob=hdi_prob)

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


def _load_rho(data, samples_df, normal=False, renormalise=True):
    clones = data["clones"]

    rho = samples_df[[x for x in samples_df.columns if x.startswith("rho")]]

    if not normal:
        if "normal" in clones:
            clones.remove("normal")

        rho = rho[[x for x in rho.columns if "normal" not in x]]

    rho = rho.values

    if renormalise:
        rho = rho / rho.sum(axis=1)[:, np.newaxis]

    return clones, rho
