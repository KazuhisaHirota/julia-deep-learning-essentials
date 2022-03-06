include("activation_function.jl")

struct Perceptrons
    nIn::Int
    w::Vector{Float64} # weights
end

function Perceptrons(nIn::Int)
    w = zeros(nIn)
    return Perceptrons(nIn, w)
end

function train!(model::Perceptrons, x::Vector{Float64}, t::Int, learningRate::Float64)
    # check if the data is classified correctly
    c = 0.
    for i in 1:model.nIn
        c += model.w[i] * x[i] * t
    end

    classified = 0
    if c > 0. # correct
        classified = 1
    else # wrong
        for i in 1:model.nIn
            # steepest descent method
            model.w[i] += learningRate * x[i] * t
        end
    end
    return classified
end

# calc \sigma(wx)
function predict(model::Perceptrons, x::Vector{Float64})
    preActivation = 0.
    for i in 1:model.nIn
        preActivation += model.w[i] * x[i]
    end
    return step(preActivation)
end