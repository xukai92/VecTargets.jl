Base.@kwdef struct Funnel <: ContinuousMultivariateDistribution 
    dim::Int=2
end

dim(funnel::Funnel) = funnel.dim

function _logpdf(::Funnel, θ::AbstractVecOrMat)
    nu, x = θ[1,:], θ[2:end,:]
    
    # lp = logpdf.(Ref(Normal(0, 3)), nu) + 
    #     logpdf.(MvNormal.(Ref(zeros(size(x, 1))), exp.(nu ./ 2)), eachcol(x))
    
#     lp = logpdf(BroadcastedNormalStd([0], [3]), nu) +
#         dropdims(sum(logpdf(BroadcastedNormalStd([0], exp.(nu ./ 2)), x); dims=1); dims=1)
    
#     return lp
    
    # diff = nu
    # s = 3
    # lp1 = -(log(2pi) .+ 2log.(s) .+ diff .* diff ./ s.^2) / 2
    lp1 = _logpdf_normal_std.(nu, 0, 3)

    # diff = x
    # s = exp.(nu ./ 2)
    # lp2 = -(log(2pi) .+ 2log.(s) .+ diff .* diff ./ s.^2) / 2
    lp2 = _logpdf_normal_std.(x, 0, exp.(nu ./ 2))

    return lp1 + dropdims(sum(lp2; dims=1); dims=1)
end

logpdf(funnel::Funnel, θ::AbstractVector) = only(_logpdf(funnel, θ))

logpdf(funnel::Funnel, θ::AbstractMatrix) = _logpdf(funnel, θ)
