import os


def setup_julia_module():
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

    import juliacall

    jl = juliacall.newmodule("cfClone")

    jl.seval("using ADTypes, Bijectors, BridgeStan, Distributions, DynamicPPL, JSON, Pigeons, ReverseDiff")

    jl.seval(
        """
    struct CfCloneDescription
    end
    """
    )

    return jl
