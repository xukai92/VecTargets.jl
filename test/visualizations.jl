using VecTargets, Plots, BSON

function contour_density(target, x, y)
    X = repeat(reshape(x, 1, :), length(y), 1)
    Y = repeat(y, 1, length(x))
    θ = hcat(X[:], Y[:])'
    Z = exp.(logpdf(target, θ))

    return contour(x, y, Z, size=(300, 300), aspect_ratio=:equal)
end

## Banana

target = Banana()

x = -4.5:0.01:4.5
y = -1.0:0.01:8.0
p = contour_density(target, x, y)
savefig(p, "banana.png")

## Gaussians

target = HighDimGaussian(2)

x = -3.0:0.01:3.0
y = -3.0:0.01:3.0
p = contour_density(target, x, y)
savefig(p, "2d_gaussian.png")

## Mixture of Gaussians

### 1D

target = OneDimGaussianMixtures()

x = -3.0:0.01:3.0
p = plot(x, exp.(logpdf(target, x')), size=(300, 300), label=nothing)
savefig(p, "1d_mog.png")

### 2D

target = TwoDimGaussianMixtures()

x = -3.0:0.01:3.0
y = -3.0:0.01:3.0
p = contour_density(target, x, y)
savefig(p, "2d_mog.png")

### Spiral

target = Spiral(50, 0.04)

x = -0.6:0.01:0.6
y = -0.6:0.01:0.6
p = contour_density(target, x, y)
savefig(p, "spiral.png")

## Finnish pine saplings dataset

datadir = joinpath(splitdir(@__DIR__)[1:end-1]..., "data")

### Raw

finpines = BSON.load(joinpath(datadir, "finpines-raw.bson"))

p = scatter(finpines[:x], finpines[:y], size=(300, 300), label=nothing, aspect_ratio=:equal)
savefig(p, "finpine-raw.png")

### Grid

ps = map([16, 32, 64]) do ngrid
    finpines_grid = BSON.load(joinpath(datadir, "finpines-$ngrid.bson"))
    data_counts = finpines_grid[:data_counts]
    heatmap(reshape(data_counts, ngrid, ngrid), aspect_ratio=:equal, title="$ngrid x $ngrid")
end
p = plot(ps..., layout=@layout([a b c]), size=(300 * 3, 300))
savefig(p, "finpine-grid.png")
