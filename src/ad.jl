using ReverseDiff: ReverseDiff, DiffResults
using LinearAlgebra: dot

function gen_grad(func, x::AbstractVector)
    f_tape = ReverseDiff.GradientTape(func, x)
    compiled_f_tape = ReverseDiff.compile(f_tape)
    result = DiffResults.GradientResult(x)
    function grad(x::AbstractVector)
        ReverseDiff.gradient!(result, compiled_f_tape, x)
        return DiffResults.value(result), DiffResults.gradient(result)
    end
    return grad
end

function gen_grad(func, x::AbstractMatrix)
    local retval
    function retval_sum(x)
        retval = func(x)
        return sum(retval)
    end
    f_tape = ReverseDiff.GradientTape(retval_sum, x)
    compiled_f_tape = ReverseDiff.compile(f_tape)
    result = DiffResults.GradientResult(x)
    function grad(x::AbstractMatrix)
        ReverseDiff.gradient!(result, compiled_f_tape, x)
        return ReverseDiff.value(retval), DiffResults.gradient(result)
    end
    return grad
end

function gen_hess(func, x::AbstractVector)
    f_tape = ReverseDiff.HessianTape(func, x)
    compiled_f_tape = ReverseDiff.compile(f_tape)
    result = DiffResults.HessianResult(x)
    function hess(x::AbstractVector)
        ReverseDiff.hessian!(result, compiled_f_tape, x)
        return DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
    end
    return hess
end

combine_hess_results(r1, r2) = map(zip(r1, r2, 1:3)) do (x1, x2, dim)
    cat(x1, x2; dims=dim)
end

function gen_hess(func, X::AbstractMatrix)
    x = first(eachcol(X))
    f_tape = ReverseDiff.HessianTape(func, x)
    compiled_f_tape = ReverseDiff.compile(f_tape)
    result = DiffResults.HessianResult(x)
    function hess(X::AbstractMatrix)
        mapreduce(combine_hess_results, eachcol(X)) do x
            ReverseDiff.hessian!(result, compiled_f_tape, x)
            DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
        end
    end
    return hess
end

function gen_Hvp(func, x::AbstractVector, v)
    grad = VecTargets.gen_grad(func, ReverseDiff.track.(x))
    # TODO Can we ignore tracking v to save computation?
    fHvp = (x, v) -> dot(grad(x)[2], v)
    f_tape = ReverseDiff.GradientTape(fHvp, (x, v))
    compiled_f_tape = ReverseDiff.compile(f_tape)
    result = DiffResults.GradientResult.((x, v))
    function Hvp(x::AbstractVector, v)
        ReverseDiff.gradient!(result, compiled_f_tape, (x, v))
        return first(DiffResults.gradient.(result)) # only returns grad
    end
    return Hvp
end

# Helper function for compiling gradient and hessian function using ReverseDiff.jl
gen_logpdf(target) = x -> logpdf(target, x)
gen_logpdf_grad(target, x) = gen_grad(gen_logpdf(target), x)
    logpdf_grad(target, x) = gen_logpdf_grad(target, x)(x)
gen_logpdf_hess(target, x) = gen_hess(gen_logpdf(target), x)
    logpdf_hess(target, x) = gen_logpdf_hess(target, x)(x)
gen_logpdf_Hvp(target, x, v) = gen_Hvp(gen_logpdf(target), x, v)
    logpdf_Hvp(target, x, v) = gen_logpdf_Hvp(target, x, v)(x, v)
