module VecTargets

using BSON, Parameters, Distributions, DistributionsAD
using Random: AbstractRNG, GLOBAL_RNG, rand!, randn!, shuffle
include("bnormal.jl")
using StatsFuns: logsumexp, logistic
using ReverseDiff: ReverseDiff, DiffResults

import Distributions: rand, logpdf, pdf

function get_∂ℓπ∂θ_reversediff(ℓπ, θ::AbstractVector)
    inputs = (θ,)
    f_tape = ReverseDiff.GradientTape(ℓπ, inputs)
    compiled_f_tape = ReverseDiff.compile(f_tape)
    results = similar.(inputs)
    all_results = DiffResults.GradientResult.(results)
    # cfg = ReverseDiff.GradientConfig(inputs)    # we may use this in the future; see https://github.com/cJuliaDiff/ReverseDiff.jl/blob/master/examples/gradient.jl#L43
    function ∂ℓπ∂θ(θ::AbstractVector)
        ReverseDiff.gradient!(all_results, compiled_f_tape, (θ,))
        return DiffResults.value(first(all_results)), DiffResults.gradient(first(all_results))
    end
    return ∂ℓπ∂θ
end

function get_∂ℓπ∂θ_reversediff(ℓπ, θ::AbstractMatrix)
    local logdensities
    function ℓπ_sum(x)
        logdensities = ℓπ(x)
        return sum(logdensities)
    end
    inputs = (θ,)
    f_tape = ReverseDiff.GradientTape(ℓπ_sum, inputs)
    compiled_f_tape = ReverseDiff.compile(f_tape)
    results = similar.(inputs)
    all_results = DiffResults.GradientResult.(results)
    # cfg = ReverseDiff.GradientConfig(inputs)    # we may use this in the future; see https://github.com/JuliaDiff/ReverseDiff.jl/blob/master/examples/gradient.jl#L43
    function ∂ℓπ∂θ(θ::AbstractMatrix)
        ReverseDiff.gradient!(all_results, compiled_f_tape, (θ,))
        return ReverseDiff.value(logdensities), DiffResults.gradient(first(all_results))
    end
    return ∂ℓπ∂θ
end

export get_∂ℓπ∂θ_reversediff

include("gaussian.jl")
export HighDimGaussian

include("banana.jl")
export Banana

include("gaussian_mixtures.jl")
export OneDimGaussianMixtures, TwoDimGaussianMixtures

include("spiral.jl")
export Spiral

include("logistic_regression.jl")
export LogisticRegression

include("coxprocess.jl")
export LogGaussianCoxPointProcess

export get_target

end # module
