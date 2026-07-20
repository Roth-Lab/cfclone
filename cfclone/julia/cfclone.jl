using ADTypes,
    AdvancedHMC,
    Bijectors,
    BridgeStan,
    CairoMakie,
    Distributions,
    ForwardDiff,
    LinearAlgebra,
    InferenceReport,
    JSON,
    LogDensityProblems,
    LogDensityProblemsAD,
    MCMCChains,
    PDMats,
    Pigeons,
    PythonCall,
    Random,
    StanLogDensityProblems

import LogDensityProblems: dimension, logdensity, logdensity_and_gradient, logdensity_gradient_and_hessian

stan_model_path(outlier::Bool) = (@__DIR__) * "/../stan/cfclone" * (outlier ? "_outlier" : "") * ".stan"

# Target setup
struct CfCloneDescription{V}
    variational_approximation::V
end

CfCloneDescription() = CfCloneDescription(nothing)

function build_target(data, stan_model_file; initialize_at_map=true, laplace_max_iters=100, optimizer_options...)
    converted_data = convert_data(data)
    description = if laplace_max_iters > 0
        CfCloneDescription(
            laplace_approximation(
                converted_data,
                stan_model_file;
                initialize_at_map,
                max_n_iters=laplace_max_iters,
                optimizer_options...,
            ),
        )
    else
        CfCloneDescription()
    end
    return StanLogPotential(stan_model_file, converted_data, description)
end

convert_data(data::String) = data # already a path to .json or JSON
convert_data(data) = JSON.json(data) # e.g. Dict

# Setup PT
function get_inputs(target; checkpoint=false, multithreaded=false, n_chains=4, n_rounds=10, seed=0)
    Inputs(;
        checkpoint,
        explorer=PreconditionedNUTS(target),
        multithreaded,
        n_chains,
        n_rounds,
        record=[traces, Pigeons.round_trip, Pigeons.timing_extrema, Pigeons.energy_ac1, Pigeons.explorer_acceptance_pr],
        reference=DistributionLogPotential(laplace_approximation(target).distribution),
        seed,
        target,
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

#### Laplace approximation 
struct LaplaceApproximation{D,T}
    distribution::D
    optimization_trace::T
    log_z_estimate::Float64
    initialize_at_map::Bool
end

laplace_approximation(target::StanLogPotential)::LaplaceApproximation =
    target.extra_information.variational_approximation

function initialization(approx::LaplaceApproximation, rng)
    approx.initialize_at_map ? copy(approx.distribution.μ) : rand(rng, approx.distribution)
end

function Pigeons.initialization(
    target::StanLogPotential{M,S,D,CfCloneDescription{A}}, rng::AbstractRNG, replica_index::Int
) where {M,S,D,A<:LaplaceApproximation}
    Pigeons.StanState(
        initialization(target.extra_information.variational_approximation, rng),
        StanRNG(target.model, rand(rng, UInt32)),
    )
end

function stan_problem(stan_model_file, data)
    model = StanModel(stan_model_file, data; warn=false, make_args=["STAN_THREADS=true", "BRIDGESTAN_AD_HESSIAN=true"])
    return StanProblem(model; nan_on_error=true)
end

Base.show(io::IO, ::StanModel) = print(io, "StanModel(...)")

function laplace_approximation(data::String, stan_model_file::String; args...)
    # Using StanLogDensityProblems here instead of Pigeons because we 
    # currently have not done the Hessian "plumbing" in Pigeons 
    laplace_approximation(stan_problem(stan_model_file, data); args...)
end

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
function log_density_with_constants(log_density_problem::StanProblem, point)
    BridgeStan.log_density(log_density_problem.model, point; propto=false)
end

#### Creating reports for the Laplace Approximation
"""
Usage:

```
target = build_target("../../data/data.json", "../stan/cfclone_outlier_model.so")
ls = laplace_samples(target)
laplace_report(ls)
```
"""
function laplace_report(ls; args...)
    report(
        ls;
        max_dim=50,
        target_name="Laplace Approximation",
        postprocessors=[InferenceReport.default_postprocessors(); laplace_diagnostics],
        args...,
    )
end

function laplace_report_with_exec(ls; exec_folder=nothing, args...)
    report(
        ls;
        max_dim=50,
        target_name="Laplace Approximation",
        postprocessors=[InferenceReport.default_postprocessors(); laplace_diagnostics],
        exec_folder=exec_folder,
        args...,
    )
end

# use this to create reports from Laplace
struct LaplaceSamples{A<:LaplaceApproximation,C}
    approximation::A
    chains::C
end

function laplace_samples(stan_log_potential; args...)
    laplace_samples(stan_log_potential.model, stan_log_potential.extra_information.variational_approximation; args...)
end

function laplace_samples(
    model::StanModel,
    laplace_approximation::LaplaceApproximation;
    n_samples=10000,
    constrained=true,
    rng=MersenneTwister(1),
)
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

    InferenceReport.add_markdown(
        context;
        title="Evidence estimate",
        contents="Laplace approximation of the evidence: ``$(laplace_approximation.log_z_estimate)``",
    )

    let
        fig = Figure(;)
        ax = Axis(fig[1, 1])
        lines!(ax, laplace_approximation.optimization_trace.values)
        name = "opt_trace"
        file = InferenceReport.output_file(context, name, "png")
        CairoMakie.save(file, fig; size=(800, 800), px_per_unit=2)
        InferenceReport.add_plot(
            context;
            title="Optimization trace",
            description="Objective function (log density) as a function of the optimizer iteration.",
            file="$name.png",
        )
    end

    let
        fig = Figure(;)
        ax = Axis(fig[1, 1]; yscale=log2)
        lines!(ax, laplace_approximation.optimization_trace.step_sizes)
        name = "step_sizes"
        file = InferenceReport.output_file(context, name, "png")
        CairoMakie.save(file, fig; size=(800, 800), px_per_unit=2)
        InferenceReport.add_plot(
            context;
            title="Optimization stepsizes",
            description="Optimizer step size as a function of the iteration.",
            file="$name.png",
        )
    end

    e = eigen(laplace_approximation.distribution.Σ)

    let
        fig = Figure(;)
        ax = Axis(fig[1, 1]; yscale=log10)
        lines!(ax, e.values)
        name = "spectrum"
        file = InferenceReport.output_file(context, name, "png")
        CairoMakie.save(file, fig; size=(800, 800), px_per_unit=2)
        InferenceReport.add_plot(
            context; title="Spectrum", file="$name.png", description="""
                                                           Sorted eigenvalues of the Laplace approximation's covariance 
                                                           matrix (in log scale). 
                                                           """
        )
    end

    let
        fig = Figure(;)
        ax = Axis(fig[1, 1])
        hm = heatmap!(ax, e.vectors)
        Colorbar(fig[1, 2], hm; label="Loading")
        name = "vectors"
        file = InferenceReport.output_file(context, name, "png")
        CairoMakie.save(file, fig; size=(800, 800), px_per_unit=2)
        InferenceReport.add_plot(
            context;
            title="Eigenvectors",
            file="$name.png",
            description="""
              Eigenvectors (columns) of the Laplace approximation's covariance 
              matrix. 
              """,
        )
    end
end

function InferenceReport.Inference(laplace::LaplaceSamples, max_dim::Int)
    InferenceReport.Inference(laplace.approximation, InferenceReport.truncate_if_needed(laplace.chains, max_dim))
end

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

function Pigeons.explorer_recorder_builders(::PreconditionedNUTS)
    [Pigeons.buffers, Pigeons.ad_buffers, Pigeons.explorer_acceptance_pr, Pigeons.explorer_n_steps]
end

LogDensityProblems.capabilities(::Pigeons.InterpolatedAD) = LogDensityProblems.LogDensityOrder{1}()

function PreconditionedNUTS(target::StanLogPotential; step_size=nothing, n_adapt=1000, rng=MersenneTwister(1))
    approx = laplace_approximation(target)
    step_size = tune_step_size(step_size, target::StanLogPotential, approx, n_adapt, rng)
    return PreconditionedNUTS(step_size, approx.distribution.Σ)
end

function tune_step_size(::Nothing, target::StanLogPotential, approx::LaplaceApproximation, n_adapt, rng)
    @info "Finding a stepsize for NUTS"
    sampler = nuts_tuning_sampler(rng, StanProblem(target.model; nan_on_error=true), approx)
    _, stats = AdvancedHMC.sample(
        rng, sampler.hamiltonian, sampler.kernel, sampler.init, n_adapt, sampler.adaptor, n_adapt; progress=true
    )
    return stats[end].step_size
end

tune_step_size(step_size, args...) = step_size

function simple_mcmc(target::StanLogPotential; n_chains=1, args...)
    pigeons(;
        target,
        n_chains,
        reference=target,
        record=[traces, Pigeons.timing_extrema, Pigeons.energy_ac1, Pigeons.explorer_acceptance_pr],
        args...,
    )
end

function Pigeons.step!(explorer::PreconditionedNUTS, replica, shared)
    state = replica.state.unconstrained_parameters
    rng = replica.rng
    log_potential = Pigeons.find_log_potential(replica, shared.tempering, shared)
    log_potential_autodiff = ADgradient(AutoForwardDiff(), log_potential, replica)
    sampler = nuts_fixed_sampler(log_potential_autodiff, explorer.step_size, explorer.inverse_mass_matrix, state)
    samples, stats = AdvancedHMC.sample(
        rng, sampler.hamiltonian, sampler.kernel, sampler.init, 2, sampler.adaptor, 0; progress=false, verbose=false
    )
    Pigeons.@record_if_requested!(
        replica.recorders, :explorer_acceptance_pr, (replica.chain, stats[end].acceptance_rate)
    )
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
function auto_non_convex_newton(target; max_n_iters=100, start_point=nothing, silent=false, eigen_cutoff=1e-5)
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
    return valid_points[best_valid_index], valid_values[best_valid_index], valid_next_step_size[best_valid_index]
end

# Armijo condition
function is_valid_proposal(step_size, value, proposed_value, gradient_norm_squared)
    # NB: seems important to decrease 10^-4 to 10^-8 for some non-convex problems (cfclone)
    proposed_value ≥ value + 10^-8 * step_size * gradient_norm_squared
end

# in a well-conditioned Gaussian case, this will return the inverse covariance matrix (= negative Hessian)
# for non-convex or ill-conditioned problems, it will construct a positive definite approximation heuristically
function compute_inverse_metric(target, point, eigen_cutoff=1e-5, silent=true)
    _, _, hessian = logdensity_gradient_and_hessian(target, point)
    neg_hess = Hermitian(-hessian)
    neg_hess = fix_neg_hessian(neg_hess)
    eigm = eigmin(neg_hess)
    silent || @info "Smallest eig value of negative Hessian = $eigm"
    return PDMat(eigm < eigen_cutoff ? neg_hess + (eigen_cutoff + 2.0 * abs(eigm)) * I : neg_hess)
end

function fix_neg_hessian(neg_hess::LinearAlgebra.Hermitian)
    if any(isnan, neg_hess)
        @warn "NaNs found in negative Hessian ... correcting"
        replace!(parent(neg_hess), NaN => 0.0)
    end
    return neg_hess
end