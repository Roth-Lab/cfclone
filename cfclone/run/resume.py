import enum
import os
import random

import h5py
import numpy as np
import pandas as pd

from cfclone.inference import run_inference
from cfclone.julia import set_env_variables, setup_julia_module
from cfclone.models import get_model


def resume(exec_dir, fit_file, out_file, num_rounds, num_threads):
    set_env_variables()

    os.environ["PYTHON_JULIACALL_THREADS"] = f"{num_threads}"

    jl = setup_julia_module()

    print("\nUsing {} threads\n".format(jl.Threads.nthreads()))

    pt = jl.seval(
        """
    function resume(exec_dir, num_rounds)
        pt = pigeons(increment_n_rounds!(exec_dir, num_rounds))
    end
    """
    )(exec_dir, num_rounds)

    with h5py.File(out_file, "w") as fh:
        with h5py.File(fit_file, "r") as fit_fh:
            fit_fh.copy(fit_fh["/data"], fh["/data"])

        dset = fh.create_dataset("/results/samples", data=samples_df.to_numpy(), dtype=np.float64)
        dset.attrs["columns"] = [str(x) for x in samples_df.columns]

        dset = fh.create_dataset("/results/summary", data=summary_df.to_numpy(), dtype=np.float64)
        dset.attrs["columns"] = [str(x) for x in summary_df.columns]

    jl.seval(
        """
    function build_report(exec_folder, pt; interval_probability=0.95)
        report(pt; exec_folder,  interval_probability, view=false)
    end
    """
    )(exec_dir, pt)
