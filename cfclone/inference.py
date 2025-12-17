def run_inference(jl, model, seed, exec_dir=None, num_chains=12, num_chains_vi=5, num_rounds=10, num_threads=1):
    get_inputs = jl.seval(
        """
    function get_inputs(reference, target; checkpoint=false, multithreaded=false, n_chains=12, n_chains_variational=5, n_rounds=10, seed=0)
        return Inputs(
            checkpoint=checkpoint,
            explorer=AutoMALA(),
            multithreaded=multithreaded,
            n_chains=n_chains,
            n_chains_variational=n_chains_variational,
            n_rounds=n_rounds,
            record=[
                traces,
                Pigeons.round_trip,
                Pigeons.timing_extrema,
                Pigeons.energy_ac1,
                Pigeons.explorer_acceptance_pr
            ],
            reference=reference,
            seed=seed,
            target=target,
            variational=GaussianReference(first_tuning_round=5)
        )
    end
        """
    )

    inputs = get_inputs(
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
        infer = jl.seval(
            """
        function infer(inputs)
            pt = PT(inputs)
            result = pigeons(pt)
            return result
        end
        """
        )

        pt = infer(inputs)

    else:
        infer = jl.seval(
            """
        function infer(exec_folder, inputs)
            pt = PT(inputs; exec_folder)
            result = pigeons(pt)
            return result
        end
        """
        )

        pt = infer(exec_dir, inputs)

    return pt
