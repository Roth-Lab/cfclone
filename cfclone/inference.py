def run_inference(jl, model, num_chains=12, num_chains_vi=5, num_rounds=10):
    infer = jl.seval(
        """
    function infer(reference, target, n_chains=12, n_chains_variational=5, n_rounds=10)
        result = pigeons(
            ;
            target,
            multithreaded = true,
            record=[traces, Pigeons.round_trip, Pigeons.timing_extrema, Pigeons.energy_ac1],
            explorer=AutoMALA(),
            n_chains,
            reference,
            n_rounds,
            n_chains_variational=5,
            variational=GaussianReference(first_tuning_round=5)
        )
        return result
    end
    """
    )

    return infer(
        model.reference,
        model.target,
        num_chains,
        num_chains_vi,
        num_rounds,
    )
