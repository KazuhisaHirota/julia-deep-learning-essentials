using Random

function makeDataset(startIndex::Int, endIndex::Int, mu1::Float64, mu2::Float64, answer::Int, X::Matrix, t::Vector, rng::MersenneTwister)
    for i in startIndex:endIndex # not endIndex - 1
        X[i, 1] = randn(rng) + mu1 # not X[i][0]
        X[i, 2] = randn(rng) + mu2 # not X[i][1]
        t[i] = answer
    end
end

function makeDataset(startIndex::Int, endIndex::Int, mu1::Float64, mu2::Float64, answer::Vector, X::Matrix, T::Matrix, rng::MersenneTwister)
    for i in startIndex:endIndex # not endIndex - 1
        X[i, 1] = randn(rng) + mu1 # not X[i][0]
        X[i, 2] = randn(rng) + mu2 # not X[i][1]
        T[i, :] = answer
    end
end