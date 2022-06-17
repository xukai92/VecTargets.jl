Base.@kwdef struct Funnel <: ContinuousMultivariateDistribution 
    dim::Int=2
end

dim(funnel::Funnel) = funnel.dim

function _logpdf(::Funnel, θ::AbstractVecOrMat)
    nu, x = θ[1,:], θ[2:end,:]
    lp = logpdf.(Ref(Normal(0, 3)), nu) + logpdf.(MvNormal.(Ref(zeros(size(x, 1))), exp.(nu ./ 2)), eachcol(x))
    return lp
end

logpdf(funnel::Funnel, θ::AbstractVector) = only(_logpdf(funnel, θ))

logpdf(funnel::Funnel, θ::AbstractMatrix) = _logpdf(funnel, θ)
