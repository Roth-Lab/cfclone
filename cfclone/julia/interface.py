from collections import namedtuple

import enum
import os
import importlib.resources

import cfclone.julia
import cfclone.stan


class SliceSamplingType(enum.Enum):
    disable = enum.auto()
    compose = enum.auto()
    mixture = enum.auto()
    only = enum.auto()


Model = namedtuple("Model", ["reference", "target"])


def get_model(jl, data, use_outlier=True):
    stan_dir = importlib.resources.files(cfclone.stan)

    if use_outlier:
        stan_file = stan_dir.joinpath("cfclone_outlier.stan")

        reference = jl.build_outlier_reference(data["num_clones"])

    else:
        stan_file = stan_dir.joinpath("cfclone.stan")

        reference = jl.build_reference(data["num_clones"])

    target = jl.build_target(data, str(stan_file))

    return Model(reference, target)


def run_inference(
    jl,
    data,
    seed,
    exec_dir=None,
    num_chains=12,
    num_chains_vi=5,
    num_rounds=10,
    num_threads=1,
    slice_sampling=SliceSamplingType.disable,
    use_outlier=False,
):

    # match slice_sampling:
    #     case SliceSamplingType.disable:
    #         explorer = jl.seval("AutoMALA()")

    #     case SliceSamplingType.compose:
    #         explorer = jl.seval("Compose(AutoMALA(), SliceSampler())")

    #     case SliceSamplingType.mixture:
    #         explorer = jl.seval("Mix(AutoMALA(), SliceSampler())")

    #     case SliceSamplingType.only:
    #         explorer = jl.seval("SliceSampler()")

    explorer = jl.seval("Mix(ClonePairReweightSampler(), SliceSampler())")

    # explorer = jl.seval("SliceSampler()")

    model = get_model(jl, data, use_outlier=use_outlier)

    inputs = jl.get_inputs(
        explorer,
        model.reference,
        model.target,
        checkpoint=(exec_dir is not None),
        multithreaded=(num_threads > 1),
        n_chains=num_chains,
        n_chains_variational=num_chains_vi,
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
