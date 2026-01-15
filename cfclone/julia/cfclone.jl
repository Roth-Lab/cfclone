using ADTypes,
    Bijectors,
    BridgeStan,
    Distributions,
    LinearAlgebra,
    InferenceReport,
    JSON,
    MCMCChains,
    Pigeons,
    Random, 
    ADTypes,
    ForwardDiff, 
    InferenceReport, 
    MCMCChains,
    StanLogDensityProblems,
    PDMats, 
    CairoMakie,
    LogDensityProblems,
    LogDensityProblemsAD,
    AdvancedHMC

import LogDensityProblems: dimension, logdensity, logdensity_and_gradient, logdensity_gradient_and_hessian

stan_model_path(outlier::Bool) = (@__DIR__) * "/../stan/cfclone" * (outlier ? "_outlier" : "") * ".stan"

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
struct CfCloneDescription{V}
    variational_approximation::V
end 
CfCloneDescription() = CfCloneDescription(nothing)

function build_target(data, stan_model_file; initialize_at_map = true, laplace_max_iters = 100, optimizer_options...)
    converted_data = convert_data(data)
    description = laplace_max_iters > 0 ? 
        try
            CfCloneDescription(laplace_approximation(converted_data, stan_model_file; initialize_at_map, max_n_iters = laplace_max_iters, optimizer_options...))
        catch e 
            @warn "Laplace approximation failed"
            CfCloneDescription()
        end : CfCloneDescription()
    return StanLogPotential(stan_model_file, converted_data, description)
end

convert_data(data::String) = data # already a path to .json or JSON
convert_data(data) = JSON.json(data) # e.g. Dict

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

get_nuts_inputs(
    target;
    checkpoint = false,
    multithreaded = false,
    n_chains = 4,
    n_rounds = 10,
    seed = 0) =
    Inputs(;
        checkpoint,
        explorer = PreconditionedNUTS(target),
        multithreaded,
        n_chains,
        n_rounds,
        record=[traces, Pigeons.round_trip, Pigeons.timing_extrema, Pigeons.energy_ac1, Pigeons.explorer_acceptance_pr],
        reference = DistributionLogPotential(laplace_approximation(target).distribution),
        seed,
        target
    )


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


#### Laplace approximation 

struct LaplaceApproximation{D, T}
    distribution::D 
    optimization_trace::T
    log_z_estimate::Float64 
    initialize_at_map::Bool
end 
laplace_approximation(target::StanLogPotential)::LaplaceApproximation = target.extra_information.variational_approximation

initialization(approx::LaplaceApproximation, rng) = approx.initialize_at_map ? 
        copy(approx.distribution.μ) :
        rand(rng, approx.distribution)
Pigeons.initialization(target::StanLogPotential{M, S, D, CfCloneDescription{A}}, rng::AbstractRNG, replica_index::Int) where {M, S, D, A <: LaplaceApproximation} =
    Pigeons.StanState(initialization(target.extra_information.variational_approximation, rng), StanRNG(target.model, rand(rng, UInt32)))


function stan_problem(stan_model_file, data)
    model = StanModel(
        stan_model_file, data, 
        warn = false, 
        make_args = ["STAN_THREADS=true", "BRIDGESTAN_AD_HESSIAN=true"])
    return StanProblem(model, nan_on_error = true)
end
Base.show(io::IO, ::StanModel) = print(io, "StanModel(...)")

laplace_approximation(data::String, stan_model_file::String; args...) = 
    # Using StanLogDensityProblems here instead of Pigeons because we 
    # currently have not done the Hessian "plumbing" in Pigeons 
    laplace_approximation(stan_problem(stan_model_file, data); args...)

function laplace_approximation(log_density_problem; initialize_at_map::Bool, optimizer_options...)
    optimizer_trace = auto_non_convex_newton(log_density_problem; optimizer_options...) 
    center = optimizer_trace.points[end] 
    covariance_approx = inv(compute_inverse_metric(log_density_problem, center))
    normal_approx = MvNormal(center, covariance_approx)
    log_z_estimate = laplace_log_z_estimate(log_density_problem, normal_approx)
    return LaplaceApproximation(normal_approx, optimizer_trace, log_z_estimate, initialize_at_map)
end

function laplace_log_z_estimate(log_density_problem, normal_approx)
    log_density_at_center = log_density_with_constants(log_density_problem, normal_approx.μ)
    return log_density_at_center + 0.5 * (logdet(normal_approx.Σ) + dimension(log_density_problem) * log(2π))
end

log_density_with_constants(log_density_problem, point) = logdensity(log_density_problem, point)
# Take into account that by default Stan skips constants, but we want to include them here:
log_density_with_constants(log_density_problem::StanProblem, point) = 
    BridgeStan.log_density(log_density_problem.model, point; propto = false) 


#### Creating reports for the Laplace Approximation

"""
Usage:

```
target = build_target("../../data/data.json", "../stan/cfclone_outlier_model.so")
ls = laplace_samples(target)
laplace_report(ls)
```
"""
laplace_report(ls; args...) = report(ls; max_dim = 50, target_name = "Laplace Approximation", postprocessors = [InferenceReport.default_postprocessors(); laplace_diagnostics], args...)

# use this to create reports from Laplace
struct LaplaceSamples{A <: LaplaceApproximation, C}
    approximation::A 
    chains::C 
end

laplace_samples(stan_log_potential; args...) = laplace_samples(stan_log_potential.model, stan_log_potential.extra_information.variational_approximation; args...)

function laplace_samples(model::StanModel, laplace_approximation::LaplaceApproximation; n_samples = 10000, constrained = true, rng = MersenneTwister(1))
    names = constrained ? BridgeStan.param_names(model) : BridgeStan.param_unc_names(model)
    array = zeros(n_samples, length(names), 1)
    for i in 1:n_samples
        unc_draw = rand(rng, laplace_approximation.distribution) 
        array[i, :, 1] = constrained ? BridgeStan.param_constrain(model, unc_draw) : unc_draw
    end
    return LaplaceSamples(laplace_approximation, Chains(array, names))
end

function laplace_diagnostics(context)
    laplace_approximation::LaplaceApproximation = context.inference.algorithm 

    InferenceReport.add_markdown(context;
        title = "Evidence estimate",
        contents = "Laplace approximation of the evidence: ``$(laplace_approximation.log_z_estimate)``"
    )

    let
        fig = Figure(;)
        ax = Axis(fig[1,1])
        lines!(ax, laplace_approximation.optimization_trace.values) 
        name = "opt_trace"
        file = InferenceReport.output_file(context, name, "png")
        CairoMakie.save(file, fig, size= (800, 800), px_per_unit=2)
        InferenceReport.add_plot(context;
            title = "Optimization trace",
            description = "Objective function (log density) as a function of the optimizer iteration.",
            file = "$name.png")
    end
    
    let 
        fig = Figure(;)
        ax = Axis(fig[1,1], yscale = log2)
        lines!(ax, laplace_approximation.optimization_trace.step_sizes) 
        name = "step_sizes"
        file = InferenceReport.output_file(context, name, "png")
        CairoMakie.save(file, fig, size= (800, 800), px_per_unit=2)
        InferenceReport.add_plot(context;
            title = "Optimization stepsizes",
            description = "Optimizer step size as a function of the iteration.",
            file = "$name.png")
    end

    e = eigen(laplace_approximation.distribution.Σ)

    let 
        fig = Figure(;)
        ax = Axis(fig[1,1], yscale = log10)
        lines!(ax, e.values) 
        name = "spectrum"
        file = InferenceReport.output_file(context, name, "png")
        CairoMakie.save(file, fig, size= (800, 800), px_per_unit=2)
        InferenceReport.add_plot(context;
            title = "Spectrum",
            file = "$name.png", 
            description = """
                Sorted eigenvalues of the Laplace approximation's covariance 
                matrix (in log scale). 
                """)
    end

    let 
        fig = Figure(;)
        ax = Axis(fig[1,1])
        hm = heatmap!(ax, e.vectors) 
        Colorbar(fig[1, 2], hm, label = "Loading")
        name = "vectors"
        file = InferenceReport.output_file(context, name, "png")
        CairoMakie.save(file, fig, size= (800, 800), px_per_unit=2)
        InferenceReport.add_plot(context;
            title = "Eigenvectors",
            file = "$name.png", 
            description = """
                Eigenvectors (columns) of the Laplace approximation's covariance 
                matrix. 
                """)
    end
end

InferenceReport.Inference(laplace::LaplaceSamples, max_dim::Int) = InferenceReport.Inference(laplace.approximation, InferenceReport.truncate_if_needed(laplace.chains, max_dim)) 


#### Preconditioned NUTS samplers 

"""
Usage: to learn the NUTS step size, use
```
target = build_target("../../data/data.json", "../stan/cfclone_outlier_model.so")
nuts = PreconditionedNUTS(target)
```
For a given step size, use 
```
nuts = PreconditionedNUTS(target, step_size = 0.1)
```

Quick and dirty test:
```
simple_mcmc(target, explorer = nuts, n_rounds = 5)
```
"""
struct PreconditionedNUTS{M} 
    step_size::Float64 
    inverse_mass_matrix::M 
end 
Pigeons.explorer_recorder_builders(::PreconditionedNUTS) = [Pigeons.buffers, Pigeons.ad_buffers, Pigeons.explorer_acceptance_pr, Pigeons.explorer_n_steps]
LogDensityProblems.capabilities(::Pigeons.InterpolatedAD) = LogDensityProblems.LogDensityOrder{1}()

function PreconditionedNUTS(target::StanLogPotential; step_size = nothing, n_adapt = 1000, rng = MersenneTwister(1))
    approx = laplace_approximation(target)
    step_size = tune_step_size(step_size, target::StanLogPotential, approx, n_adapt, rng)
    return PreconditionedNUTS(step_size, approx.distribution.Σ)
end
function tune_step_size(::Nothing, target::StanLogPotential, approx::LaplaceApproximation, n_adapt, rng)
    @info "Finding a stepsize for NUTS"
    sampler = nuts_tuning_sampler(rng, StanProblem(target.model, nan_on_error = true), approx) 
    _, stats = AdvancedHMC.sample(
        rng, sampler.hamiltonian, sampler.kernel, sampler.init, n_adapt, sampler.adaptor, n_adapt; progress = true
    )
    return stats[end].step_size 
end
tune_step_size(step_size, args...) = step_size

simple_mcmc(target::StanLogPotential; n_chains = 1, args...) =
    pigeons(; 
        target, 
        n_chains,
        reference = target, 
        record=[traces, Pigeons.timing_extrema, Pigeons.energy_ac1, Pigeons.explorer_acceptance_pr],
        args...)

function Pigeons.step!(explorer::PreconditionedNUTS, replica, shared)
    state = replica.state.unconstrained_parameters
    rng = replica.rng
    log_potential = Pigeons.find_log_potential(replica, shared.tempering, shared)
    log_potential_autodiff = ADgradient(AutoForwardDiff(), log_potential, replica)
    sampler = nuts_fixed_sampler(log_potential_autodiff, explorer.step_size, explorer.inverse_mass_matrix, state)
    samples, stats = AdvancedHMC.sample(
        rng, sampler.hamiltonian, sampler.kernel, sampler.init, 2, sampler.adaptor, 0; progress = false, verbose = false
    )
    Pigeons.@record_if_requested!(replica.recorders, :explorer_acceptance_pr, (replica.chain, stats[end].acceptance_rate))
    Pigeons.@record_if_requested!(replica.recorders, :explorer_n_steps, (replica.chain, stats[end].n_steps))
    state .= samples[end]
end

function nuts_tuning_sampler(rng, log_density_problem, laplace_approximation::LaplaceApproximation)
    init = initialization(laplace_approximation, rng)
    inverse_mass_matrix = laplace_approximation.distribution.Σ
    metric = DenseEuclideanMetric(inverse_mass_matrix)
    hamiltonian = Hamiltonian(metric, log_density_problem)
    integrator = Leapfrog(find_good_stepsize(hamiltonian, init))
    kernel = HMCKernel(Trajectory{SliceTS}(integrator, GeneralisedNoUTurn()))    
    adaptor = StepSizeAdaptor(0.8, integrator)
    return (; kernel, hamiltonian, adaptor, init)
end

function nuts_fixed_sampler(log_density_problem, step_size, inverse_mass_matrix::AbstractMatrix, current_point)
    init = current_point
    metric = DenseEuclideanMetric(inverse_mass_matrix)
    hamiltonian = Hamiltonian(metric, log_density_problem)
    integrator = Leapfrog(step_size)
    kernel = HMCKernel(Trajectory{SliceTS}(integrator, GeneralisedNoUTurn()))    
    adaptor = NoAdaptation()
    return (; kernel, hamiltonian, adaptor, init)
end


#### Second order optimizer robust to non-convexity 

"""
A straightforward extension of https://arxiv.org/pdf/2510.09923 into a second order method. 

The `target` should conform the `LogDensityProblemsAD` interface.
"""
function auto_non_convex_newton(target; max_n_iters = 100, start_point = nothing, silent = false, eigen_cutoff = 1e-5)
    step_size = 1.0
    d = dimension(target)
    point = start_point === nothing ? zeros(d) : start_point

    value = logdensity(target, point)
    @assert isfinite(value)

    points = [point] 
    values = [value]
    step_sizes = [step_size]

    silent || @info "auto_non_convex_newton started. Initial value = $value"

    for iter in 1:max_n_iters 
        if converged(target, points, values, step_sizes, silent)
            break 
        end
        point, value, step_size = auto_non_convex_newton_iteration(target, point, step_size, eigen_cutoff, silent)
        
        push!(points, point)
        push!(values, value)
        push!(step_sizes, step_size) 

        silent || @info "Value at iteration $iter = $value (step size = $step_size)" 
    end

    return (; points, values, step_sizes)
end

function converged(target, points, values, step_sizes, silent)
    _, gradient = logdensity_and_gradient(target, points[end])
    gradient_inf_norm = norm(gradient, Inf)
    silent || @info "gradient infinity norm = $gradient_inf_norm"
    return gradient_inf_norm < 1e-8 # default in Optim.jl 
end

function auto_non_convex_newton_iteration(target, point, step_size, eigen_cutoff, silent)
    value, gradient = logdensity_and_gradient(target, point)
    gradient_norm_squared = sum(abs2, gradient)  
    
    valid_next_step_size = [step_size / 4] 
    valid_values = [value]
    valid_points = [point]

    inverse_metric = compute_inverse_metric(target, point, eigen_cutoff, silent)
    base_update = inverse_metric \ gradient
    for proposed_step_size in [step_size / 2, step_size, step_size * 2]
        proposed_point = point + proposed_step_size * base_update
        proposed_value = logdensity(target, proposed_point)
        if is_valid_proposal(proposed_step_size, value, proposed_value, gradient_norm_squared)
            push!(valid_next_step_size, proposed_step_size) 
            push!(valid_values, proposed_value)
            push!(valid_points, proposed_point)
        end
    end
    best_valid_index = argmax(valid_values)
    silent || @info "Selected auto index (1 to stay, 2 shrink, 3 to keep, 4 to grow): $best_valid_index"
    return  valid_points[best_valid_index], 
            valid_values[best_valid_index],
            valid_next_step_size[best_valid_index]
end

# Armijo condition
function is_valid_proposal(step_size, value, proposed_value, gradient_norm_squared) 
    # NB: seems important to decrease 10^-4 to 10^-8 for some non-convex problems (cfclone)
    proposed_value ≥ value + 10^-8 * step_size * gradient_norm_squared
end

# in a well-conditioned Gaussian case, this will return the inverse covariance matrix (= negative Hessian)
# for non-convex or ill-conditioned problems, it will construct a positive definite approximation heuristically
function compute_inverse_metric(target, point, eigen_cutoff = 1e-5, silent = true)
    _, _, hessian = logdensity_gradient_and_hessian(target, point)
    neg_hess = Hermitian(-hessian) 
    eigm = eigmin(neg_hess) 
    silent || @info "Smallest eig value of negative Hessian = $eigm"
    return PDMat(eigm < eigen_cutoff ? neg_hess + (eigen_cutoff + 2. * abs(eigm)) * I : neg_hess)
end