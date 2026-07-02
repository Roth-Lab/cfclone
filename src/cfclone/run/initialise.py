import numpy as np

from cfclone.julia import run_inference, setup_julia_module


def initialise():
    data = _get_toy_data()

    jl = setup_julia_module()

    pt = run_inference(
        jl,
        data,
        0,
        num_chains=5,
        num_rounds=2,
        use_outlier=False,
    )

    pt = run_inference(
        jl,
        data,
        0,
        num_chains=5,
        num_rounds=2,
        use_outlier=True,
    )


def _get_toy_data():
    M = 20

    K = 2

    cn_a = np.random.randint(1, 6, size=(K, M))

    cn_b = np.random.randint(1, 6, size=(K, M))

    cn_t = cn_a + cn_b

    a = np.random.randint(1, 100, size=M)

    b = np.random.randint(1, 100, size=M)

    d = a + b

    rdr = np.random.normal(0, 1, size=M)

    data = {
        "num_clones": K,
        "num_bins": M,
        "cn_a": cn_a,
        "cn_t": cn_t,
        "a": a,
        "d": d,
        "rdr": rdr,
        "pi": np.ones(K),
    }

    return data
