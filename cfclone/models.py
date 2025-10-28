from collections import namedtuple

import importlib.resources

import cfclone.stan


Model = namedtuple("Model", ["reference", "target"])


def get_model(jl, data, use_outlier=True):
    stan_dir = importlib.resources.files(cfclone.stan)

    if use_outlier:
        stan_file = stan_dir.joinpath("cfclone_outlier.stan")

        build_reference = jl.seval(
            """
        function build_reference(n_clones)
            result = []
            for _ in 1:(n_clones - 1)
                push!(result, Uniform(0, 1)) # rho
            end
            append!(result,
                    [Distributions.Gamma(1, 1),     # alpha
                    Distributions.Beta(1, 100),     # non_binomiality
                    Distributions.Gamma(1, 10),    # sigma
                    Distributions.Gamma(1, 1),      # sigma_outlier 
                    Distributions.Gamma(1, 100),    # outlier_rate_rdr
                    Distributions.Gamma(1, 100)]    # outlier_rate_baf
            )
            return DistributionLogPotential(product_distribution(transformed.(result)))
        end
        """
        )

    else:
        stan_file = stan_dir.joinpath("cfclone.stan")

        build_reference = jl.seval(
            """
        function build_reference(n_clones)
            result = []
            for _ in 1:(n_clones - 1)
                push!(result, Uniform(0, 1)) # rho
            end
            append!(result,
                    [Distributions.Gamma(1, 1),     # alpha
                    Distributions.Beta(1, 100),     # non_binomiality
                    Distributions.Gamma(1, 100)]    # sigma
            )
            return DistributionLogPotential(product_distribution(transformed.(result)))
        end
        """
        )

    build_target = jl.seval(
        """
    function build_target(data, stan_model_file)
        description = CfCloneDescription() 
        return StanLogPotential(stan_model_file, JSON.json(data), description)
    end
    """
    )

    reference = build_reference(data["num_clones"])

    target = build_target(data, str(stan_file))

    return Model(reference, target)
