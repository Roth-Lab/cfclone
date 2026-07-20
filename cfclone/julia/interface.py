import os
import importlib.resources

import cfclone.julia
import cfclone.stan


def get_target(jl, data, use_outlier=True):
    stan_dir = importlib.resources.files(cfclone.stan)

    if use_outlier:
        stan_file = stan_dir.joinpath("cfclone_outlier.stan")

    else:
        stan_file = stan_dir.joinpath("cfclone.stan")

    target = jl.build_target(data, str(stan_file))

    return target


def run_inference(
    jl,
    data,
    seed,
    exec_dir=None,
    laplace_exec_dir=None,
    num_chains=12,
    num_rounds=10,
    num_threads=1,
    use_outlier=False,
):
    target = get_target(jl, data, use_outlier=use_outlier)
    
    if laplace_exec_dir is not None:
        ls = jl.laplace_samples(target)
        
        jl.laplace_report_with_exec(ls, exec_folder=laplace_exec_dir)

    inputs = jl.get_inputs(
        target,
        checkpoint=(exec_dir is not None),
        multithreaded=(num_threads > 1),
        n_chains=num_chains,
        n_rounds=num_rounds,
        seed=seed,
    )

    if exec_dir is None:
        pt = jl.infer_no_exec(inputs)

    else:
        pt = jl.infer_with_exec(exec_dir, inputs)

    return pt


def setup_julia_module(num_threads=1):
    set_env_variables(num_threads=num_threads)

    import juliacall

    jl = juliacall.newmodule("cfClone")

    print("\nUsing {} threads\n".format(jl.Threads.nthreads()))

    julia_file = importlib.resources.files(cfclone.julia).joinpath("cfclone.jl")

    with open(julia_file, "r") as fh:
        jl.seval("\n".join(fh.readlines()))

    return jl


def set_env_variables(num_threads=1):
    # TODO: TBB_CXX_TYPE might need to be either clang or gcc,
    #  we should have a function to grab uname to determine OS, something something MacOS
    os.environ["TBB_CXX_TYPE"] = "gcc"
    os.environ["TBB_INTERFACE_NEW"] = "new"
    os.environ["STAN_THREADS"] = "true"

    os.environ["PYTHON_JULIACALL_THREADS"] = f"{num_threads}"
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
