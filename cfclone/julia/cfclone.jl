using ADTypes,
    Bijectors,
    BridgeStan,
    Distributions,
    DynamicPPL,
    LinearAlgebra,
    InferenceReport,
    JSON,
    MCMCChains,
    Pigeons,
    PythonCall,
    Random,
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
        Distributions.Gamma(1, 100),
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
            Distributions.Beta(1, 100),
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
## Pair reweight sampler
@kwdef struct ClonePairReweightSampler
    idxs::Array{Integer}
    num_scans = 100
end

function ClonePairReweightSampler(lp::StanLogPotential)
    model = Pigeons.stan_model(lp)
    names = BridgeStan.param_names(model)
    idxs = findall(x -> occursin("rho", x), names)
    return ClonePairReweightSampler(idxs, 100)
end

function Pigeons.step!(explorer::ClonePairReweightSampler, replica, shared)
    state = replica.state
    rng = replica.rng
    log_potential = Pigeons.find_log_potential(replica, shared.tempering, shared)
    model = Pigeons.stan_model(log_potential)
    log_p = log_potential(state)
    for i in 1:explorer.num_scans
        # Propose a swap
        unc_params_old = state.unconstrained_parameters
        params = BridgeStan.param_constrain(model, unc_params_old)
        u, v = sample(rng, explorer.idxs, 2; replace=false)
        w = rand(rng, Float64)
        temp_u = params[u]
        temp_v = params[v]
        params[u] = (1 - w) * temp_u
        params[v] = w * temp_u + temp_v
        params[explorer.idxs] = params[explorer.idxs] ./ sum(params[explorer.idxs])
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

## Rho MH sampler
@kwdef struct PrevalenceRWMHSampler
    idxs::Array{Integer}
    num_dims::Integer
    num_scans = 100
    std_devs::Array{Float64}
end

function PrevalenceRWMHSampler(lp::StanLogPotential)
    model = Pigeons.stan_model(lp)
    names = BridgeStan.param_unc_names(model)
    idxs = findall(x -> occursin("rho", x), names)
    num_dims = length(idxs)
    return PrevalenceRWMHSampler(idxs, num_dims, 100, 0.1 * ones(num_dims))
end

function Pigeons.step!(explorer::PrevalenceRWMHSampler, replica, shared)
    state = replica.state
    rng = replica.rng
    log_potential = Pigeons.find_log_potential(replica, shared.tempering, shared)
    log_p = log_potential(state)
    for i in 1:explorer.num_scans
        sigma = build_preconditioner(zeros(explorer.num_dims), rng, explorer.std_devs)
        params_old = state.unconstrained_parameters[explorer.idxs]
        proposal = MvNormal(params_old, Diagonal(sigma))
        params_new = rand(rng, proposal)
        state.unconstrained_parameters[explorer.idxs] = params_new
        log_p_new = log_potential(state)
        log_q_new = logpdf(proposal, params_new)
        proposal = MvNormal(params_new, Diagonal(sigma))
        log_q_old = logpdf(proposal, params_old)
        accept_ratio = exp((log_p_new - log_q_new) - (log_p - log_q_old))
        if accept_ratio < 1 && rand(rng) > accept_ratio
            state.unconstrained_parameters[explorer.idxs] = params_old
        else
            log_p = log_p_new
        end
    end
end

function build_preconditioner(dest::Vector{T}, rng, std_devs::Vector{T}) where {T<:Real}
    @assert length(dest) == length(std_devs)
    u = rand(rng)
    if u ≤ 1 / 3
        map!(s -> iszero(s) ? 0.1 * one(T) : s, dest, std_devs)
    elseif u ≤ 2 / 3
        fill!(dest, 0.1 * one(T))
    else
        mix = rand(rng, T)
        rmix = one(T)-mix
        map!(s -> iszero(s) ? 0.1 * one(T) : mix + rmix * s, dest, std_devs)
    end
    return dest
end

function Pigeons.adapt_explorer(explorer::PrevalenceRWMHSampler, reduced_recorders, current_pt, new_tempering)
    std_devs = sqrt.(Pigeons.get_transformed_statistic(reduced_recorders, :singleton_variable, Pigeons.Variance))
    return PrevalenceRWMHSampler(explorer.idxs, explorer.num_dims, explorer.num_scans, std_devs[explorer.idxs])
end

## Slice sampler
mutable struct MySliceSampler
    idxs::Array{Integer}
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

function slice_sample!(explorer::MySliceSampler, state::AbstractVector, log_potential, cached_lp, replica)
    rng = replica.rng
    cached_lp = Pigeons.cached_log_potential(log_potential, replica.state, cached_lp) # note: we pass `replica.state` instead of `state` in case the latter is the vector version of a non-vector state (e.g. stan and dppl models)
    # iterate over coordinates
    for c in shuffle(rng, explorer.idxs)
        pointer = Ref(state, c)
        cached_lp = Pigeons.slice_sample_coord!(
            explorer.slice_sampler, replica, pointer, log_potential, cached_lp, typeof(pointer[])
        ) # note: when state is mixed, pointer is RefArray{generic common type} for all coordinates, so can't use it to dispatch 

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
    return slice_sample!(h, state.unconstrained_parameters, args...)
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
