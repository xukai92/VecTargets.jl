module VecTargets

include("bnormals.jl")
using BSON, Parameters, Distributions, DistributionsAD
using Random: GLOBAL_RNG, shuffle
using StatsFuns: logsumexp, logistic
include("ad.jl")

import Distributions: dim, rand, logpdf, pdf

include("banana.jl")
export Banana

include("funnel.jl")
export Funnel

include("high_dim_gaussian.jl")
export HighDimGaussian

include("gaussian_mixtures.jl")
export OneDimGaussianMixtures, TwoDimGaussianMixtures

include("spiral.jl")
export Spiral

include("logistic_regression.jl")
export LogisticRegression

include("coxprocess.jl")
export LogGaussianCoxPointProcess

export dim, rand, logpdf, pdf, logpdf_grad, gen_logpdf_grad, logpdf_hess, gen_logpdf_hess

end # module
