import os

# import juliapkg
import numpy as np

from cfclone.inference import run_inference
from cfclone.julia import set_env_variables, setup_julia_module
from cfclone.models import get_model


def initialise():
    set_env_variables()

    data = _get_toy_data()

    jl = setup_julia_module()

    model = get_model(jl, data, use_outlier=False)
    pt = run_inference(
        jl,
        model,
        num_chains=5,
        num_chains_vi=5,
        num_rounds=2,
    )

    model = get_model(jl, data, use_outlier=True)
    pt = run_inference(
        jl,
        model,
        num_chains=5,
        num_chains_vi=5,
        num_rounds=2,
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
