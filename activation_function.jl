function step(x::Float64)
    return x < 0. ? -1 : 1
end

function sigmoid(x::Float64)
    return 1. / (1. + exp(-x))
end

function dsigmoid(y::Float64)
    return y * (1. - y)
end

function dtanh(y::Float64)
    return 1. - y * y
end

function softmax(x::Vector{Float64})
    e = exp.(x .- maximum(x)) # to avoid overflow
    return e ./ sum(e)
end