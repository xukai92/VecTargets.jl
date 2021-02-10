# High-dimensional standard Gaussian
struct HighDimGaussian{T<:AbstractBroadcastedNormal} <: ContinuousMultivariateDistribution
    bn::T
end

dim(hdg::HighDimGaussian) = size(hdg.bn.m, 1)

HighDimGaussian(dim::Int) = HighDimGaussian(BroadcastedNormalStd(zeros(dim), ones(dim)))

logpdf(hdg::HighDimGaussian, θ::AbstractVector) = sum(logpdf(hdg.bn, θ))

logpdf(hdg::HighDimGaussian, θ::AbstractMatrix) = dropdims(sum(logpdf(hdg.bn, θ); dims=1); dims=1)

function _logpdf_grad(hdg::HighDimGaussian, x::AbstractVecOrMat{T}) where {T}
    diff = x .- hdg.bn.m
    v = -(log(2 * T(pi)) .+ logvar(hdg.bn) .+ diff .* diff ./ var(hdg.bn)) / 2
    g = -diff
    return v, g
end

function logpdf_grad(hdg::HighDimGaussian, θ::AbstractVector)
    v, g = _logpdf_grad(hdg, θ)
    return sum(v), g
end

function logpdf_grad(hdg::HighDimGaussian, θ::AbstractMatrix)
    v, g = _logpdf_grad(hdg, θ)
    return dropdims(sum(v; dims=1); dims=1), g
end

gen_logpdf_grad(hdg::HighDimGaussian, _) = x -> logpdf_grad(hdg, x)
