using MultivariateStats: MultivariateStats

function reduce_dim_by_pca(design_matrix, target_dim::Int)
    M = fit(MultivariateStats.PCA, design_matrix'; maxoutdim=target_dim)
    return Matrix(MultivariateStats.predict(M, design_matrix')')
end

struct LogisticRegression{TX, Ty, T}
    X::TX
    y::Ty
    lambda::T
end

function LogisticRegression(
    datadir::String, lambda::AbstractFloat; 
    num_obs::Int=1_000, num_latent::Int=300
)
    @unpack design_matrix, response = BSON.load(joinpath(datadir, "germancredit.bson"))
    design_matrix = size(design_matrix, 2) == num_latent ? design_matrix :
        reduce_dim_by_pca(design_matrix, num_latent)
        # design_matrix[:,1:num_latent]
    return LogisticRegression(design_matrix[1:num_obs,:], Bool.(response[1:num_obs]), lambda)
end

LogisticRegression(lambda::AbstractFloat; kwargs...) = 
    LogisticRegression(joinpath(splitdir(splitdir(pathof(@__MODULE__))[1])[1], "data"), lambda; kwargs...)

dim(lr::LogisticRegression) = size(lr.X, 2) + 2

function _logpdf(lr::LogisticRegression, theta)
    logv = theta[1,:]   # n_chains = size(theta, 2)
    v = exp.(logv)      # n_chains
    a = theta[2,:]      # n_chains
    b = theta[3:end,:]  # 300 x n_chains
    logitp = a' .+ lr.X * b
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

_logpdf_exponential(x, lambda) = log(lambda) - lambda * x 
_logpdf_bernoulli(x, p) = x * log(p) + (1 - x) * log(1 - p)

# TODO extact out some functions and define @rule for them
function _logpdf(lr::LogisticRegression, theta::AbstractVector{T}) where {T}
    logv = @view theta[1:1]
    v = exp.(logv)    # 1
    a = @view theta[2:2]    # 1
    b = @view theta[3:end]  # 300
    # lr.X: 1000 x 300
    logitp = a .+ lr.X * b # 1000 
    p = logistic.(logitp) 
    p = p * (1 - 2 * eps(T)) .+ eps(T)    # numerical stability

    logabsdetjacob = logv
    # logprior = logpdf(Exponential(1 / lr.lambda), v) + logabsdetjacob
    logprior = sum(_logpdf_exponential.(v, lr.lambda) + logabsdetjacob)
    # s = sqrt.(v)
    # logprior += logpdf(BroadcastedNormalStd(zeros(T, 1), s), a)
    logprior += sum(_logpdf_normal_var.(a, 0, v))
    # logprior_b = logpdf(BroadcastedNormalStd(zeros(T, 1, 1), s'), b)
    # logprior += dropdims(sum(logprior_b; dims=1); dims=1)
    logprior += sum(_logpdf_normal_var.(b, 0, v))
    # loglike_elementwise = logpdf.(Bernoulli.(p), lr.y)
    # loglike = dropdims(sum(loglike_elementwise; dims=1); dims=1)
    loglike = sum(_logpdf_bernoulli.(lr.y, p))
    return logprior + loglike
end

# function logpdf(lr::LogisticRegression, theta::AbstractVector)
#     theta = reshape(theta, length(theta), 1)
#     lp = _logpdf(lr, theta)
#     return only(lp)
# end

# logpdf(lr::LogisticRegression, theta::AbstractMatrix) = _logpdf(lr, theta)

logpdf(lr::LogisticRegression, theta::AbstractVecOrMat) = _logpdf(lr, theta)
