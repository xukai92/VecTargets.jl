struct Banana <: ContinuousMultivariateDistribution end

dim(::Banana) = 2

function _logpdf(::Banana, θ::AbstractVecOrMat)
    x1, x2 = θ[1,:], θ[2,:]
    U = (1 .- x1).^2 + 10(x2 - x1.^2).^2
    return -U
end

logpdf(banana::Banana, θ::AbstractVector) = only(_logpdf(banana, θ))

logpdf(banana::Banana, θ::AbstractMatrix) = _logpdf(banana, θ)

function _logpdf_grad(banana::Banana, θ::AbstractVecOrMat)
    x1, x2 = θ[1,:], θ[2,:]
    x1sq = x1.^2
    x2x1sq_diff = x2 - x1sq
    dx1 = 2(1 .- x1) + 40x2x1sq_diff .* x1
    dx2 = -20x2x1sq_diff
    return logpdf(banana, θ), cat(dx1', dx2'; dims=1)
end

function logpdf_grad(banana::Banana, θ::AbstractVector)
    v, g = _logpdf_grad(banana, θ)
    return v, dropdims(g; dims=2)
end

logpdf_grad(banana::Banana, θ::AbstractMatrix) = _logpdf_grad(banana, θ)

gen_logpdf_grad(banana::Banana, _) = x -> _logpdf_grad(banana, x)
