using ReverseDiff: ReverseDiff, DiffResults

function gen_grad(func, x::AbstractVector)
    inputs = (x,)
    f_tape = ReverseDiff.GradientTape(func, inputs)
    compiled_f_tape = ReverseDiff.compile(f_tape)
    results = similar.(inputs)
    all_results = DiffResults.GradientResult.(results)
    function grad(x::AbstractVector)
        ReverseDiff.gradient!(all_results, compiled_f_tape, (x,))
        return DiffResults.value(first(all_results)), DiffResults.gradient(first(all_results))
    end
    return grad
end

function gen_grad(func, x::AbstractMatrix)
    local retval
    function retval_sum(x)
        retval = func(x)
        return sum(retval)
    end
    inputs = (x,)
    f_tape = ReverseDiff.GradientTape(retval_sum, inputs)
    compiled_f_tape = ReverseDiff.compile(f_tape)
    results = similar.(inputs)
    all_results = DiffResults.GradientResult.(results)
    function grad(x::AbstractMatrix)
        ReverseDiff.gradient!(all_results, compiled_f_tape, (x,))
        return ReverseDiff.value(retval), DiffResults.gradient(first(all_results))
    end
    return grad
end

# Use ReverseDiff to compile gradient function
gen_logpdf_grad(target, x) = gen_grad(_x -> logpdf(target, _x), x)
logpdf_grad(target, x) = gen_grad(_x -> logpdf(target, _x), x)(x)
