function step(x::Float64)
    return x < 0. ? -1 : 1
end

function softmax(x::Vector{Float64})
    e = exp.(x .- maximum(x)) # to avoid overflow
    return e ./ sum(e)
end