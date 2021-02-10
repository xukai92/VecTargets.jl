using DrWatson
@quickactivate

finpines = wload(projectdir("finpines-raw.bson"))

# Normalize data to unit square
data_x = (finpines[:x] .+ 5) / 10
data_y = (finpines[:y] .+ 8) / 10

for ngrid in [16, 32, 64]

    grid = range(0, 1; length=(ngrid + 1))
    dimension = ngrid^2
    data_counts = zeros(Int, dimension)
    for i in 1:ngrid, j in 1:ngrid
        logical_y = (data_x .> grid[i]) .* (data_x .< grid[i+1])
        logical_x = (data_y .> grid[j]) .* (data_y .< grid[j+1])
        data_counts[(i-1)*ngrid+j] = sum(logical_y .* logical_x)
    end

    sigmasq = 1.91
    mu = log(126) - 0.5 * sigmasq
    beta = 1 / 33
    dimension = ngrid^2
    area = 1 / dimension

    data = @dict(data_counts, ngrid, dimension, sigmasq, mu, beta, area)
    wsave(projectdir("finpines-$ngrid.bson"), data)

end
