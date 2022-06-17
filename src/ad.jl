using ReverseDiff: ReverseDiff, DiffResults

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

function gen_hess(func, X::AbstractMatrix)
    x = first(eachcol(X))
    f_tape = ReverseDiff.GradientTape(func, x)
    compiled_f_tape = ReverseDiff.compile(f_tape)
    result = DiffResults.HessianResult(x)
    function hess(X::AbstractMatrix)
        mapreduce(combine_hess_results, eachcol(X)) do x
            ReverseDiff.gradient!(result, compiled_f_tape, x)
            DiffResults.value(result), DiffResults.gradient(result), DiffResults.hessian(result)
        end
    end
    return hess
end

# Use ReverseDiff to compile gradient and hessian function
gen_logpdf_grad(target, x) = gen_grad(_x -> logpdf(target, _x), x)
    logpdf_grad(target, x) = gen_logpdf_grad(target, x)(x)
gen_logpdf_hess(target, x) = gen_hess(_x -> logpdf(target, _x), x)
    logpdf_hess(target, x) = gen_logpdf_hess(target, x)(x)
