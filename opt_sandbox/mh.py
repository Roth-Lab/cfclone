from collections import namedtuple
from math import lgamma

import functools
import numba
import numpy as np
import pandas as pd
import scipy.stats as ss


Cache = namedtuple("Cache", ["c", "o_b", "o_r", "ploidy"])

Data = namedtuple("Data", ["a", "d", "rdr", "cn_a", "cn_t"])

Priors = namedtuple("Priors", ["nb_a", "nb_b", "rho", "sigma_a", "sigma_b", "w_a", "w_b"])

Parameters = namedtuple("Parameters", ["nb", "rho", "sigma", "w_b", "w_r"])

State = namedtuple("State", ["annealing_param", "cache", "data", "params", "priors"])


def main(args):
    """
    Basic MH inference for the cfClone BAF model.

    Currently assuming a Binomial data distribution.
    Focus here is updating the prevalence so not updating outlier rate.

    Note: If we go MH then a spike a slab model is possible!
    """
    rng = np.random.default_rng(1)

    bins = load_bins(args.clone_cnv_file, args.data_file)

    clones, cn_a, cn_t = load_clone_cnv_data(bins, args.clone_cnv_file)

    a, d, rdr = load_data(bins, args.data_file)

    data = Data(a, d, rdr, cn_a, cn_t)

    cache = load_cache(data)

    num_clones = cn_t.shape[0]

    # TODO: Move prior and init param creation to functions so the model specification is
    # encapsulated outside the main loop
    # TODO: After the above refactor into a single class so the model specification is completely
    # done in the class
    priors = Priors(
        nb_a=1,
        nb_b=100,
        rho=0.5 * np.ones(num_clones),
        sigma_a=1,
        sigma_b=100,
        w_a=1,
        w_b=100,
    )

    priors.rho[-1] = 10

    params = Parameters(
        nb=rng.beta(priors.nb_a, priors.nb_b),
        rho=rng.dirichlet(priors.rho),
        sigma=rng.gamma(priors.sigma_a, (1 / priors.sigma_b)),
        w_b=rng.beta(priors.w_a, priors.w_b),
        w_r=rng.beta(priors.w_a, priors.w_b),
    )

    # Using partial functions here to use random walk with prior and uniform proposal
    nb_updates = [
        functools.partial(update_non_binomality_random_walk, use_prior_proposal=True),
        functools.partial(update_non_binomality_random_walk, use_prior_proposal=False),
    ]

    rho_updates = [
        functools.partial(update_rho_random_walk, use_prior_proposal=True),
        functools.partial(update_rho_random_walk, use_prior_proposal=False),
        update_rho_move_weight,
        update_rho_swap_weight,
    ]

    outlier_baf_updates = [
        functools.partial(update_baf_outlier_rate_random_walk, use_prior_proposal=True),
        functools.partial(update_baf_outlier_rate_random_walk, use_prior_proposal=False),
    ]

    outlier_rdr_updates = [
        functools.partial(update_rdr_outlier_rate_random_walk, use_prior_proposal=True),
        functools.partial(update_rdr_outlier_rate_random_walk, use_prior_proposal=False),
    ]

    num_iters = int(2e5)

    state = State(0, cache, data, params, priors)

    for i in range(int(num_iters)):
        # Set below 1 to obtain the sampler with no annealing
        # TODO: Move the annealing scheduling to a function or class
        # TODO: Check reversible PT paper, is ^3 or ^(1/3) prefered?
        annealing_param = min(1, (i / (num_iters / 2)) ** 3)

        # annealing_param = 1

        state = State(annealing_param, state.cache, state.data, state.params, state.priors)

        # TODO: This is not needed when annealing is off.
        # The previous cached version can be used for efficiency
        log_p = log_joint(state)

        # TODO: Need to add the term to account for the proposal in these updates
        # TODO: Updates could be made adaptive based on previous accept rates. Will want this
        # for a Julia/Pigeons implementation.
        updates = [
            rng.choice(nb_updates),
            rng.choice(rho_updates),
            rng.choice(outlier_baf_updates),
            rng.choice(outlier_rdr_updates),
            update_sigma_random_walk,
        ]

        rng.shuffle(updates)

        # Note: It might be better to just randomly choose one update here
        for f in updates:
            log_p, state = f(log_p, state, rng)

        if i % 10000 == 0:
            print(i, state.annealing_param, log_p, log_joint(state, annealing_param=1))

            print(
                state.params.rho.sum(),
                state.params.rho[:-1].sum(),
                state.params.nb,
                state.params.sigma,
                state.params.w_b,
                state.params.w_r,
            )

            print()

            for c, r in zip(clones, state.params.rho):
                print(c, r)

            print()


def load_cache(data):
    num_bins = len(data.a)

    # Sum of log factorials from BetaBinomial pdf
    c = 0

    # BAF outlier pmf
    o_b = np.zeros(num_bins)

    # RDR outleir pmf
    o_r = np.zeros(num_bins)

    for i in range(num_bins):
        c += log_binomial_coefficient(data.d[i], data.a[i])

        o_b[i] = log_beta_binomial_likelihood(data.d[i], data.a[i], 1, 1)

        o_r[i] = log_student_t_pdf(data.rdr[i], 4, 1, 1)

    ploidy = data.cn_t.mean(axis=1)

    return Cache(c, o_b, o_r, ploidy)


# MH updates
def update_rho_random_walk(log_p, state, rng, mix_rate=0.01, use_prior_proposal=True):
    """
    Attempt a random walk update centered on current rho value.

    Proposal can be uniform on the simplex or from the prior based on `use_prior_proposal` argument.
    """
    if use_prior_proposal:
        x = state.priors.rho

    else:
        x = np.ones(state.priors.rho.shape)

    rho = rng.dirichlet(x)

    state_new = _get_new_state(state, rho=rho)

    log_p_new = log_joint(state_new)

    log_q_new = ss.dirichlet.logpdf(rho, x)

    log_q_old = ss.dirichlet.logpdf(state.params.rho, x)

    accept, log_p, state = _do_mh(log_p_new, log_p, log_q_new, log_q_old, state_new, state, rng)

    return log_p, state


# def update_rho_random_walk(log_p, state, rng, mix_rate=0.01, use_prior_proposal=True):
#     """
#     Attempt a random walk update centered on current rho value.

#     Proposal can be uniform on the simplex or from the prior based on `use_prior_proposal` argument.
#     """
#     if use_prior_proposal:
#         x = state.priors.rho

#     else:
#         x = np.ones(state.priors.rho.shape)

#     rho_step = rng.dirichlet(x)

#     rho_u = mix_rate * rho_step + (1 - mix_rate) * state.params.rho

#     rho = rho_u / rho_u.sum()

#     state_new = _get_new_state(state, rho=rho)

#     log_p_new = log_joint(state_new)

#     log_q_new = ss.dirichlet.logpdf(rho_step, x)

#     rho_step_back = (state.params.rho - (1 - mix_rate) * rho_u) / mix_rate

#     print(rho_step_back)

#     print((1 - mix_rate) * rho + mix_rate * rho_step_back - state.params.rho)

#     if rho_step_back.min() < 0:
#         return log_p, state

#     log_q_old = ss.dirichlet.logpdf(rho_step_back, x)

#     accept, log_p, state = _do_mh(log_p_new, log_p, log_q_new, log_q_old, state_new, state, rng)

#     if accept:
#         print(f"Accepted random walk using prior {use_prior_proposal}")

#     return log_p, state


def update_rho_move_weight(log_p, state, rng):
    """
    Attempt to move a random amount of prevalence from one clone to another
    """
    num_clones = state.data.cn_a.shape[0]

    u, v = rng.choice(num_clones, size=2, replace=False)

    mix_rate = rng.uniform(0, 0.5)

    rho = state.params.rho.copy()

    # Move weight w from clone u to v
    rho[u] = (1 - mix_rate) * state.params.rho[u]

    rho[v] = mix_rate * state.params.rho[u] + state.params.rho[v]

    rho = rho / rho.sum()

    state_new = _get_new_state(state, rho=rho)

    log_p_new = log_joint(state_new)

    accept, log_p, state = _do_mh(log_p_new, log_p, 0, 0, state_new, state, rng)

    return log_p, state


def update_rho_swap_weight(log_p, state, rng):
    """
    Attempt to swap the prevalence of two clones
    """
    num_clones = state.data.cn_a.shape[0]

    u, v = rng.choice(num_clones, size=2, replace=False)

    rho = state.params.rho.copy()

    # Move weight w from clone u to v
    # Note: Unlike other moves we don't need to renormalise here since swapping rho[u] <-> rho[v]
    # shold not change sum(rho) due to numerical precision
    rho[u] = state.params.rho[v]

    rho[v] = state.params.rho[u]

    state_new = _get_new_state(state, rho=rho)

    log_p_new = log_joint(state_new)

    accept, log_p, state = _do_mh(log_p_new, log_p, 0, 0, state_new, state, rng)

    return log_p, state


def update_non_binomality_random_walk(log_p, state, rng, mix_rate=0.01, use_prior_proposal=True):
    """
    Attempt a random walk update centered on current outlier rate value.

    Proposal can be uniform on the simplex or from the prior based on `use_prior_proposal` argument.
    """
    if use_prior_proposal:
        x = (state.priors.nb_a, state.priors.nb_a)

    else:
        x = (1, 1)

    nb_step = rng.beta(*x)

    nb = mix_rate * nb_step + (1 - mix_rate) * state.params.nb

    state_new = _get_new_state(state, nb=nb)

    log_p_new = log_joint(state_new)

    log_q_new = ss.beta.logpdf(nb_step, *x)

    nb_step_back = (state.params.nb - (1 - mix_rate) * nb) / mix_rate

    log_q_old = ss.beta.logpdf(nb_step_back, *x)

    accept, log_p, state = _do_mh(log_p_new, log_p, log_q_new, log_q_old, state_new, state, rng)

    return log_p, state


def update_baf_outlier_rate_random_walk(log_p, state, rng, mix_rate=0.01, use_prior_proposal=True):
    """
    Attempt a random walk update centered on current outlier rate value.

    Proposal can be uniform on the simplex or from the prior based on `use_prior_proposal` argument.
    """
    if use_prior_proposal:
        x = (state.priors.w_a, state.priors.w_b)

    else:
        x = (1, 1)

    w_step = rng.beta(*x)

    w = mix_rate * w_step + (1 - mix_rate) * state.params.w_b

    state_new = _get_new_state(state, w_b=w)

    log_p_new = log_joint(state_new)

    log_q_new = ss.beta.logpdf(w_step, *x)

    w_step_back = (state.params.w_b - (1 - mix_rate) * w) / mix_rate

    log_q_old = ss.beta.logpdf(w_step_back, *x)

    accept, log_p, state = _do_mh(log_p_new, log_p, log_q_new, log_q_old, state_new, state, rng)

    return log_p, state


def update_rdr_outlier_rate_random_walk(log_p, state, rng, mix_rate=0.01, use_prior_proposal=True):
    """
    Attempt a random walk update centered on current outlier rate value.

    Proposal can be uniform on the simplex or from the prior based on `use_prior_proposal` argument.
    """
    if use_prior_proposal:
        x = (state.priors.w_a, state.priors.w_b)

    else:
        x = (1, 1)

    w_step = rng.beta(*x)

    w = mix_rate * w_step + (1 - mix_rate) * state.params.w_r

    state_new = _get_new_state(state, w_r=w)

    log_p_new = log_joint(state_new)

    log_q_new = ss.beta.logpdf(w_step, *x)

    w_step_back = (state.params.w_r - (1 - mix_rate) * w) / mix_rate

    log_q_old = ss.beta.logpdf(w_step_back, *x)

    accept, log_p, state = _do_mh(log_p_new, log_p, log_q_new, log_q_old, state_new, state, rng)

    return log_p, state


def update_sigma_random_walk(log_p, state, rng, mix_rate=0.01, precision=100, use_prior_proposal=True):
    """
    Attempt a random walk update centered on current rho value.

    Proposal can be uniform on the simplex or from the prior based on `use_prior_proposal` argument.
    """

    def get_standard_params(x, s):
        b = x * s
        a = b * x
        return a, b

    a, b = get_standard_params(state.params.sigma, precision)

    sigma = rng.gamma(a, 1 / b)

    if sigma < 1e-100:
        sigma = 1e-100

    state_new = _get_new_state(state, sigma=sigma)

    log_p_new = log_joint(state_new)

    log_q_new = ss.gamma.logpdf(sigma, a, scale=1 / b)

    a, b = get_standard_params(sigma, precision)

    log_q_old = ss.gamma.logpdf(state.params.sigma, a, scale=1 / b)

    accept, log_p, state = _do_mh(log_p_new, log_p, log_q_new, log_q_old, state_new, state, rng)

    return log_p, state


def _do_mh(log_p_new, log_p_old, log_q_new, log_q_old, state_new, state_old, rng):
    u = rng.random()

    diff = (log_p_new - log_q_new) - (log_p_old - log_q_old)

    if np.isinf(diff):
        return False, log_p_old, state_old

    if np.log(u) < diff:
        accept = True

        log_p = log_p_new

        state = state_new

    else:
        accept = False

        log_p = log_p_old

        state = state_old

    return accept, log_p, state


def _get_new_state(state, nb=None, rho=None, sigma=None, w_b=None, w_r=None):
    params_new = {}

    if nb is None:
        params_new["nb"] = state.params.nb
    else:
        params_new["nb"] = nb

    if rho is None:
        params_new["rho"] = state.params.rho
    else:
        params_new["rho"] = rho

    if sigma is None:
        params_new["sigma"] = state.params.sigma
    else:
        params_new["sigma"] = sigma

    if w_b is None:
        params_new["w_b"] = state.params.w_b
    else:
        params_new["w_b"] = w_b

    if w_r is None:
        params_new["w_r"] = state.params.w_r
    else:
        params_new["w_r"] = w_r

    return State(
        annealing_param=state.annealing_param,
        cache=state.cache,
        data=state.data,
        params=Parameters(**params_new),
        priors=state.priors,
    )


# Model specification
def log_joint(state, annealing_param=None):
    if annealing_param is None:
        annealing_param = state.annealing_param

    return log_prior(state.params, state.priors) + annealing_param * log_likelihood(
        state.cache, state.data, state.params
    )


def log_prior(params, priors):
    log_p = 0
    log_p += ss.beta.logpdf(params.nb, priors.nb_a, priors.nb_b)
    log_p += ss.dirichlet.logpdf(params.rho, priors.rho)
    log_p += ss.beta.logpdf(params.w_b, priors.w_a, priors.w_b)
    log_p += ss.beta.logpdf(params.w_r, priors.w_a, priors.w_b)
    log_p += ss.gamma.logpdf(params.sigma, priors.sigma_a, scale=(1 / priors.sigma_b))
    return log_p


# def log_likelihood(cache, data, params):
#     p = (params.rho @ data.cn_a) / (params.rho @ data.cn_t)
#     return _log_likelihood(cache, data.a, data.d, p, params.w)


# @numba.njit(parallel=True)
# def _log_likelihood(cache, a, d, p, w):
#     # Note: The factorial term is pre-computed thus log_likelihoods used for BAF data
#     log_p = cache.c
#     N = len(a)
#     lmw_0 = np.log(w)
#     lmw_1 = np.log1p(-w)
#     temp = np.zeros((N, 2))
#     for i in numba.prange(N):
#         temp[i, 0] = lmw_0 + cache.o[i]
#         temp[i, 1] = lmw_1 + log_binomial_likelihood(d[i], a[i], p[i])
#     for i in numba.prange(N):
#         log_p += log_sum_exp(temp[i])
#     return log_p


def log_likelihood(cache, data, params):
    return log_likelihood_baf(cache, data, params) + log_likelihood_rdr(cache, data, params)


def log_likelihood_baf(cache, data, params):
    p = (params.rho @ data.cn_a) / (params.rho @ data.cn_t)
    a = p / params.nb
    b = (1 - p) / params.nb
    return _log_likelihood_baf(cache, data.a, data.d, a, b, params.w_b)


def log_likelihood_rdr(cache, data, params):
    mu = (params.rho @ data.cn_a) / (params.rho @ cache.ploidy)
    return _log_likelihood_rdr(cache, data.rdr, mu, params.sigma, params.w_r)


@numba.njit(parallel=True)
def _log_likelihood_baf(cache, a, d, nb_a, nb_b, w):
    # Note: The factorial term is pre-computed thus log_likelihoods used for BAF data
    log_p = cache.c
    N = len(a)
    lmw_0 = np.log(w)
    lmw_1 = np.log1p(-w)
    temp = np.zeros((N, 2))
    for i in numba.prange(N):
        temp[i, 0] = lmw_0 + cache.o_b[i]
        temp[i, 1] = lmw_1 + log_beta_binomial_likelihood(d[i], a[i], nb_a[i], nb_b[i])
    for i in numba.prange(N):
        log_p += log_sum_exp(temp[i])
    return log_p


@numba.njit(parallel=True)
def _log_likelihood_rdr(cache, data, mu, sigma, w):
    # Note: The factorial term is pre-computed thus log_likelihoods used for BAF data
    log_p = 0
    N = len(data)
    lmw_0 = np.log(w)
    lmw_1 = np.log1p(-w)
    temp = np.zeros((N, 2))
    log_norm = log_student_t_inv_norm_const(25, sigma)
    for i in numba.prange(N):
        temp[i, 0] = lmw_0 + cache.o_r[i]
        temp[i, 1] = lmw_1 + log_norm + log_student_t_likelihood(data[i], 25, mu[i], sigma)
    for i in numba.prange(N):
        log_p += log_sum_exp(temp[i])
    return log_p


# Math functions wrapped in Numba for speed
@numba.njit("float64(float64[:])")
def log_sum_exp(log_X):
    max_exp = log_X.max()
    if np.isinf(max_exp):
        return max_exp
    total = 0.0
    for x in log_X:
        total += np.exp(x - max_exp)
    return np.log(total) + max_exp


@numba.vectorize()
def log_gamma(x):
    return lgamma(x)


@numba.njit
def log_beta(a, b):
    if a <= 0 or b <= 0:
        return -np.inf
    return log_gamma(a) + log_gamma(b) - log_gamma(a + b)


@numba.njit
def log_factorial(x):
    return log_gamma(x + 1)


@numba.njit
def log_binomial_coefficient(n, x):
    return log_factorial(n) - log_factorial(x) - log_factorial(n - x)


@numba.njit
def log_factorial(x):
    return log_gamma(x + 1)


@numba.njit
def log_beta_binomial_likelihood(n, x, a, b):
    return log_beta(a + x, b + n - x) - log_beta(a, b)


@numba.njit
def log_binomial_likelihood(n, x, p):
    if p == 0:
        if x == 0:
            return 0
        else:
            return -np.inf
    if p == 1:
        if x == n:
            return 0
        else:
            return -np.inf
    return x * np.log(p) + (n - x) * np.log(1 - p)


@numba.njit
def log_binomial_pdf(n, x, p):
    return log_binomial_coefficient(n, x) + log_binomial_likelihood(n, x, p)


@numba.njit
def log_beta_binomial_pdf(n, x, a, b):
    return log_binomial_coefficient(n, x) + log_beta_binomial_likelihood(n, x, a, b)


@numba.njit
def log_student_t_likelihood(x, nu, mu, sigma):
    return -0.5 * (nu + 1) * np.log(1 + (1 / nu) * (((x - mu) / sigma) ** 2))


@numba.njit
def log_student_t_inv_norm_const(nu, sigma):
    return log_gamma(0.5 * (nu + 1)) - log_gamma(0.5 * nu) - 0.5 * np.log(nu * np.pi) - np.log(sigma)


@numba.njit
def log_student_t_pdf(x, nu, mu, sigma):
    return log_student_t_inv_norm_const(nu, sigma) + log_student_t_likelihood(x, nu, mu, sigma)


# Data loading code
def load_bins(clone_cnv_file, data_file):
    df = pd.read_csv(data_file, sep="\t")
    _add_bin_name_col(df)
    clone_df = pd.read_csv(clone_cnv_file, converters={"clone": str}, sep="\t")
    _add_bin_name_col(clone_df)
    # Ensure the same set of bins is used and data is aligned
    bins = pd.merge(
        df[["bin_name"]].drop_duplicates(), clone_df[["bin_name"]].drop_duplicates(), on="bin_name", how="inner"
    )["bin_name"]
    return bins


def load_clone_cnv_data(bins, file_name):
    df = pd.read_csv(file_name, converters={"clone": str}, sep="\t")
    _add_bin_name_col(df)
    cn_a = df.pivot(index="clone", columns="bin_name", values="cn_a")[bins]
    cn_b = df.pivot(index="clone", columns="bin_name", values="cn_b")[bins]
    cn_a = _add_normal_clone(cn_a)
    cn_b = _add_normal_clone(cn_b)
    cn_a.loc["normal", [x for x in cn_a.columns if x.split(":")[0].replace("chr", "") == "Y"]] = 0
    cn_b.loc["normal", [x for x in cn_b.columns if x.split(":")[0].replace("chr", "") == "Y"]] = 0
    cn_t = cn_a + cn_b
    print(f"Analysing using {cn_t.shape[0]} clones and {cn_t.shape[1]} bins")
    return list(cn_a.index), cn_a.to_numpy(), cn_t.to_numpy()


def load_data(bins, file_name):
    df = pd.read_csv(file_name, sep="\t")
    _add_bin_name_col(df)
    a = df.set_index("bin_name").loc[bins, "a"]
    b = df.set_index("bin_name").loc[bins, "b"]
    d = a + b
    rdr = df.set_index("bin_name").loc[bins, "rdr"]
    return a.to_numpy(), d.to_numpy(), rdr.to_numpy()


def _add_bin_name_col(df):
    df["bin_name"] = df["chrom"].astype(str) + ":" + df["start"].astype(str) + ":" + df["end"].astype(str)


def _add_normal_clone(df):
    clones = list(df.index)
    clones.append("normal")
    bins = df.columns
    vals = df.to_numpy()
    vals = np.vstack([vals, np.ones(df.shape[1])])
    return pd.DataFrame(vals, index=clones, columns=bins)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--clone-cnv-file", required=True)

    parser.add_argument("-d", "--data-file", required=True)

    cli_args = parser.parse_args()

    main(cli_args)
