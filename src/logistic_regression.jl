struct LogisticRegression{TX, Ty, T}
    X::TX
    y::Ty
    lambda::T
end

function LogisticRegression(datadir::String, lambda::AbstractFloat)
    @unpack design_matrix, response = BSON.load(joinpath(datadir, "germancredit.bson"))
    return LogisticRegression(design_matrix', response, lambda)
end

LogisticRegression(lambda::AbstractFloat) = 
    LogisticRegression(joinpath(splitdir(@__FILE__)[1:end-2]..., "data"), lambda)

dim(lr::LogisticRegression) = size(lr.X, 1) + 2

function _logpdf(lr::LogisticRegression, theta)
    n_chains = size(theta, 2)
    logv = theta[1,:]
    v = exp.(logv)      # n_chains
    a = theta[2,:]      # n_chains
    b = theta[3:end,:]  # 300 x n_chains
    logitp = a' .+ lr.X' * b
    p = logistic.(logitp)
    p = p * (1 - 2 * eps()) .+ eps()    # numerical stability

    logabsdetjacob = logv
    logprior = logpdf.(Ref(Exponential(1 / lr.lambda)), v) + logabsdetjacob
    s = sqrt.(v)
    T = eltype(s)
    logprior += logpdf(BroadcastedNormalStd(zeros(T, 1), s), a)
    logprior_b = logpdf(BroadcastedNormalStd(zeros(T, 1, 1), s'), b)
    logprior += dropdims(sum(logprior_b; dims=1); dims=1)
    loglike_elementwise = logpdf.(Bernoulli.(p), lr.y)
    loglike = dropdims(sum(loglike_elementwise; dims=1); dims=1)
    return logprior + loglike
end

function logpdf(lr::LogisticRegression, theta::AbstractVector)
    theta = reshape(theta, length(theta), 1)
    lp = _logpdf(lr, theta)
    return only(lp)
end

logpdf(lr::LogisticRegression, theta::AbstractMatrix) = _logpdf(lr, theta)
