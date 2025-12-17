import os


def set_env_variables():
    # TODO: TBB_CXX_TYPE might need to be either clang or gcc,
    #  we should have a function to grab uname to determine OS, something something MacOS
    os.environ["TBB_CXX_TYPE"] = "gcc"
    os.environ["TBB_INTERFACE_NEW"] = "new"
    os.environ["STAN_THREADS"] = "true"


def setup_julia_module():
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

    import juliacall

    jl = juliacall.newmodule("cfClone")

    jl.seval(
        "using ADTypes, Bijectors, BridgeStan, Distributions, DynamicPPL, InferenceReport, JSON, Pigeons, ReverseDiff, MCMCChains, PythonCall"
    )

    jl.seval(
        """
    struct CfCloneDescription
    end
    """
    )

    return jl
