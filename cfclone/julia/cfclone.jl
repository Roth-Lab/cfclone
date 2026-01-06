using ADTypes,
    Bijectors,
    BridgeStan,
    Distributions,
    DynamicPPL,
    InferenceReport,
    JSON,
    MCMCChains,
    Pigeons,
    PythonCall,
    ReverseDiff

# Reference distributions
function build_reference(n_clones)
    result = []
    for _ in 1:(n_clones - 1)
        push!(result, Uniform(0, 1)) # rho
    end
    append!(result, [
        Distributions.Gamma(1, 1),      # alpha
        Distributions.Beta(1, 100),     # non_binomiality
        Distributions.Gamma(1, 100),    # sigma
    ])
    return DistributionLogPotential(product_distribution(transformed.(result)))
end

function build_outlier_reference(n_clones)
    result = []
    for _ in 1:(n_clones - 1)
        push!(result, Uniform(0, 1)) # rho
    end
    append!(
        result,
        [
            Distributions.Gamma(1, 1),      # alpha
            Distributions.Beta(1, 100),     # non_binomiality
            Distributions.Gamma(1, 100),    # sigma
            Distributions.Beta(1, 100),     # outlier_rate_baf
            Distributions.Beta(1, 100),     # outlier_rate_rdr
        ],
    )
    return DistributionLogPotential(product_distribution(transformed.(result)))
end

# Target setup
struct CfCloneDescription end

function build_target(data, stan_model_file)
    description = CfCloneDescription()
    return StanLogPotential(stan_model_file, JSON.json(data), description)
end

# Custom samplers
@kwdef struct ClonePairReweightSampler
    num_scans = 100
end

function Pigeons.step!(explorer::ClonePairReweightSampler, replica, shared)
    state = replica.state
    rng = replica.rng
    log_potential = Pigeons.find_log_potential(replica, shared.tempering, shared)
    model = Pigeons.stan_model(log_potential)
    names = BridgeStan.param_names(model)
    idxs = findall(x -> occursin("rho", x), names)
    log_p = log_potential(state)
    for i in 1:explorer.num_scans
        # Propose a swap
        unc_params_old = state.unconstrained_parameters
        params = BridgeStan.param_constrain(model, unc_params_old)
        u, v = sample(rng, idxs, 2; replace=false)
        w = rand(rng, Float64)
        temp_u = params[u]
        temp_v = params[v]
        params[u] = (1 - w) * temp_u
        params[v] = w * temp_u + temp_v
        params[idxs] = params[idxs] ./ sum(params[idxs])
        unc_params_new = BridgeStan.param_unconstrain(model, params)
        state.unconstrained_parameters = unc_params_new
        log_p_new = log_potential(state)
        accept_ratio = exp(log_p_new - log_p)
        if accept_ratio < 1 && rand(rng) > accept_ratio
            state.unconstrained_parameters = unc_params_old
        else
            log_p = log_p_new
        end
    end
end

mutable struct MySliceSampler
    idxs::Array{Int}
    slice_sampler::Pigeons.SliceSampler
end

function MySliceSampler(lp::StanLogPotential)
    model = Pigeons.stan_model(lp)
    names = BridgeStan.param_unc_names(model)
    idxs = findall(x -> !occursin("rho", x), names)
    return MySliceSampler(idxs, Pigeons.SliceSampler())
end

function Pigeons.step!(explorer::MySliceSampler, replica, shared)
    log_potential = Pigeons.find_log_potential(replica, shared.tempering, shared)
    cached_lp = -Inf
    for _ in 1:explorer.slice_sampler.n_passes
        cached_lp = slice_sample!(explorer, replica.state, log_potential, cached_lp, replica)
    end
end

function slice_sample!(h::MySliceSampler, state::AbstractVector, log_potential, cached_lp, replica)
    cached_lp = Pigeons.cached_log_potential(log_potential, replica.state, cached_lp) # note: we pass `replica.state` instead of `state` in case the latter is the vector version of a non-vector state (e.g. stan and dppl models)

    # iterate over coordinates
    for c in shuffle(h.idxs)
        pointer = Ref(state, c)
        cached_lp = Pigeons.slice_sample_coord!(h.slice_sampler, replica, pointer, log_potential, cached_lp, typeof(pointer[])) # note: when state is mixed, pointer is RefArray{generic common type} for all coordinates, so can't use it to dispatch 

        # check we still have a healthy state
        if !isfinite(cached_lp)
            error("""Got an invalid log density after updating state at index $c:
            - log density = $cached_lp
            - state[$c]   = $(pointer[])
            Dumping full replica state:
            $(replica.state)
            """)
        end
    end
    return cached_lp
end

function slice_sample!(h::MySliceSampler, state::Pigeons.StanState, args...)
    slice_sample!(h, state.unconstrained_parameters, args...)
end

# Setup PT
function get_inputs(
    explorer,
    reference,
    target;
    checkpoint=false,
    multithreaded=false,
    n_chains=12,
    n_chains_variational=5,
    n_rounds=10,
    seed=0,
)
    return Inputs(;
        checkpoint=checkpoint,
        explorer=explorer,
        multithreaded=multithreaded,
        n_chains=n_chains,
        n_chains_variational=n_chains_variational,
        n_rounds=n_rounds,
        record=[traces, Pigeons.round_trip, Pigeons.timing_extrema, Pigeons.energy_ac1, Pigeons.explorer_acceptance_pr],
        reference=reference,
        seed=seed,
        target=target,
        variational=GaussianReference(; first_tuning_round=5),
    )
end

function infer_no_exec(inputs)
    pt = PT(inputs)
    result = pigeons(pt)
    return result
end

function infer_with_exec(exec_folder, inputs)
    pt = PT(inputs; exec_folder)
    result = pigeons(pt)
    return result
end
