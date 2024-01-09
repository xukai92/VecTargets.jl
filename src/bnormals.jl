# Broadcasted Normals that works on GPUs

using Random: AbstractRNG, rand!, randn!

function _rsimilar(rng::AbstractRNG, f!::Function, x::AbstractArray, dims::Int...)
    u = similar(x, size(x)..., dims...)
    f!(rng, u)
    return u
end

rsimilar(rng, f!, x::AbstractArray, dims::Int...) = _rsimilar(rng, f!, x, dims...)

randsimilar(rng::AbstractRNG, x::AbstractArray, dims::Int...) = rsimilar(rng, rand!, x, dims...)
randnsimilar(rng::AbstractRNG, x::AbstractArray, dims::Int...) = rsimilar(rng, randn!, x, dims...)

using Distributions: Distributions, VariateForm, ValueSupport, Discrete, Continuous, Distribution, ContinuousMultivariateDistribution
import Distributions: logpdf, pdf, cdf, invlogcdf, ccdf, rand, mean, std, var, mode, minimum, maximum

struct Batch <: VariateForm end

const ContinuousBatchDistribution = Distribution{Batch,Continuous}

abstract type AbstractBroadcastedNormal <: ContinuousBatchDistribution end

function rand(rng::AbstractRNG, bd::AbstractBroadcastedNormal, dims::Int...)
    return bd.m .+ std(bd) .* randnsimilar(rng, bd.m, dims...)
end

function _logpdf_normal_std(x, m, s)
    diff = x - m
    return -(log(2π) + 2log(s) + diff * diff / s^2) / 2
end

function _logpdf_normal_var(x, m, v)
    diff = x - m
    return -(log(2π) + log(v) + diff * diff / v) / 2
end

function logpdf(bd::AbstractBroadcastedNormal, x)
    diff = x .- bd.m
    # NOTE Removed type stable conversion for π to make ReverseDiff happy
    return -(log(2 * π) .+ logvar(bd) .+ diff .* diff ./ var(bd)) / 2
end

mean(bd::AbstractBroadcastedNormal) = bd.m
mode(bd::AbstractBroadcastedNormal) = bd.m

_vlogv(bd::AbstractBroadcastedNormal) = (var(bd), logvar(bd))

function kldiv(bd1::AbstractBroadcastedNormal, bd2::AbstractBroadcastedNormal)
    diff = bd2.m .- bd1.m
    v1, logv1 = _vlogv(bd1)
    v2, logv2 = _vlogv(bd2)
    return (logv2 .- logv1 .- 1 .+ v1 ./ v2 .+ diff .* diff ./ v2) / 2
end

"""
Broadcasted Normal distribution with standard deviation.
"""
struct BroadcastedNormalStd{Tm<:AbstractArray,Ts<:AbstractArray} <: AbstractBroadcastedNormal
    m::Tm
    s::Ts
end

Broadcast.broadcastable(bd::BroadcastedNormalStd) = Ref(bd)

NormalStd(m::AbstractArray, s::AbstractArray) = BroadcastedNormalStd(m, s)

   std(bd::BroadcastedNormalStd) = bd.s
   var(bd::BroadcastedNormalStd) = bd.s.^2
logvar(bd::BroadcastedNormalStd) = 2log.(bd.s)

"""
Broadcasted Normal distribution with variance.
"""
struct BroadcastedNormalVar{Tm<:AbstractArray,Tv<:AbstractArray} <: AbstractBroadcastedNormal
    m::Tm
    v::Tv
end

Broadcast.broadcastable(bd::BroadcastedNormalVar) = Ref(bd)

NormalVar(m::AbstractArray, v::AbstractArray) = BroadcastedNormalVar(m, v)

   std(bd::BroadcastedNormalVar) = sqrt.(bd.v)
   var(bd::BroadcastedNormalVar) = bd.v
logvar(bd::BroadcastedNormalVar) = log.(bd.v)

"""
Broadcasted Normal distribution with log standard deviation.
"""
struct BroadcastedNormalLogStd{Tm<:AbstractArray,Ts<:AbstractArray} <: AbstractBroadcastedNormal
       m::Tm    # mean
    logs::Ts    # log std
end

Broadcast.broadcastable(bd::BroadcastedNormalLogStd) = Ref(bd)

NormalLogStd(m::AbstractArray, logs::AbstractArray) = BroadcastedNormalLogStd(m, logs)

   std(bd::BroadcastedNormalLogStd) = exp.(bd.logs)
   var(bd::BroadcastedNormalLogStd) = exp.(2bd.logs)
logvar(bd::BroadcastedNormalLogStd) = 2bd.logs

function _vlogv(bd::BroadcastedNormalLogStd)
    logv = logvar(bd)
    v = exp.(logv)
    return v, logv
end

"""
Broadcasted Normal distribution with log variance.
"""
struct BroadcastedNormalLogVar{Tm<:AbstractArray,Tv<:AbstractArray} <: AbstractBroadcastedNormal
       m::Tm    # mean
    logv::Tv    # log std
end

Broadcast.broadcastable(bd::BroadcastedNormalLogVar) = Ref(bd)

NormalLogVar(m::AbstractArray, logv::AbstractArray) = BroadcastedNormalLogVar(m, logv)

   std(bd::BroadcastedNormalLogVar) = exp.(bd.logv ./ 2)
   var(bd::BroadcastedNormalLogVar) = exp.(bd.logv)
logvar(bd::BroadcastedNormalLogVar) = bd.logv

function _vlogv(bd::BroadcastedNormalLogVar)
    logv = logvar(bd)
    v = exp.(logv)
    return v, logv
end
