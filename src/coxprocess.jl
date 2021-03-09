struct LogGaussianCoxPointProcess{T<:AbstractVector{Int}, F<:AbstractFloat, D}
    counts::T
    dimension::Int
    area::F
    prior_X::D
end

# Ref: https://github.com/pierrejacob/debiasedhmc/blob/master/inst/coxprocess/model.R#L6-L22
function LogGaussianCoxPointProcess(datadir::String, ngrid::Int)
    @unpack data_counts, ngrid, dimension, sigmasq, mu, beta, area = 
        BSON.load(joinpath(datadir, "finpines-$ngrid.bson"))
    μ = fill(mu, dimension)
    Σ = Matrix{Float64}(undef, dimension, dimension)
    for m in 1:dimension, n in 1:dimension
        i = [floor(Int, (m - 1) / ngrid) + 1, (m - 1) % ngrid + 1]
        j = [floor(Int, (n - 1) / ngrid) + 1, (n - 1) % ngrid + 1]
        Σ[m,n] = sigmasq * exp(-sqrt(sum((i - j).^2)) / (ngrid * beta))
    end
    prior_X = MvNormal(μ, Σ)
    return LogGaussianCoxPointProcess(data_counts, dimension, area, prior_X)
end

LogGaussianCoxPointProcess(ngrid::Int) = 
    LogGaussianCoxPointProcess(joinpath(splitdir(@__FILE__)[1:end-2]..., "data"), ngrid)

dim(lgcpp::LogGaussianCoxPointProcess) = lgcpp.dimension

function loglikelihood(lgcpp::LogGaussianCoxPointProcess, x::AbstractVector)
    cumsum = 0
    for i in 1:length(x)
        cumsum += x[i] * lgcpp.counts[i] - lgcpp.area * exp(x[i])
    end
    return cumsum
end

function loglikelihood(lgcpp::LogGaussianCoxPointProcess, x::AbstractMatrix)
    n_chains = size(x, 2)
    return map(n -> loglikelihood(lgcpp, x[:,n]), 1:n_chains)
end

_logpdf(lgcpp::LogGaussianCoxPointProcess, x) = logpdf(lgcpp.prior_X, x) + loglikelihood(lgcpp, x)

function logpdf(lgcpp::LogGaussianCoxPointProcess, x::AbstractVector)
    theta = reshape(x, length(x), 1)
    lp = _logpdf(lgcpp, x)
    return only(lp)
end

logpdf(lgcpp::LogGaussianCoxPointProcess, x::AbstractMatrix) = _logpdf(lgcpp, x)
