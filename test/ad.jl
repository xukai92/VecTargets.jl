using Test, VecTargets
using VecTargets: gen_grad

@testset "AD Tests" begin
    for target in [Banana(), HighDimGaussian(10)]
        d = dim(target)
        for x in [randn(d), randn(d, 100)]
            v, g = logpdf_grad(target, x)
            logpdf_grad_ad = gen_grad(_x -> logpdf(target, _x), x)
            vad, gad = logpdf_grad_ad(x)
            @test v ≈ vad
            @test g ≈ gad
        end
    end
end
