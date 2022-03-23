include("activation_function.jl")

using Random

struct HiddenLayer
    nIn::Int
    nOut::Int
    W::Matrix{Float64}
    b::Vector{Float64}
    rng::MersenneTwister
    activation::Function
    dactivation::Function
end

function HiddenLayer(nIn::Int, nOut::Int, rng::MersenneTwister, activationName::String)
    # initialize W
    W = zeros(nOut, nIn)
    limit = 1. / Float64(nIn)
    for j in 1:nOut
        for i in 1:nIn
            # uniform on [-limit, limit]
            W[j, i] = limit * (2 * rand(rng) - 1)
        end
    end

    # initialize b
    b = zeros(nOut)

    if activationName == "sigmoid"
        activation = x -> sigmoid(x)
        dactivation = x -> dsigmoid(x)
    elseif activationName == "tanh"
        activation = x -> tanh(x)
        dactivation = x -> dtanh(x)
    else
        println("Error: this activation function is not supported")
        # TODO
    end

    return HiddenLayer(nIn, nOut, W, b, rng, activation, dactivation)
end

# calc \sigma(Wx + b)
function output(layer::HiddenLayer, x::Vector{Float64})
    y = zeros(layer.nOut)
    for j in 1:layer.nOut
        preActivation = 0.
        # W: (nOut, nIn), x: (nIn), b: (nOut)
        for i in 1:layer.nIn
            preActivation += layer.W[j, i] * x[i]
        end
        preActivation += layer.b[j]

        y[j] = layer.activation(preActivation)
    end
    return y
end

function forward(layer::HiddenLayer, x::Vector{Float64})
    return output(layer, x)
end

function backward!(layer::HiddenLayer, X::Matrix{Float64}, Z::Matrix{Float64},
                   dY::Matrix{Float64}, prevW::Matrix{Float64},
                   minibatchSize::Int, learningRate::Float64)
    dZ = zeros(minibatchSize, layer.nOut)
    gradW = zeros(layer.nOut, layer.nIn)
    gradb = zeros(layer.nOut)

    # train with sigmoid
    # calculate backpropagation error to get gradient of W, b
    for n in 1:minibatchSize
        for j in 1: layer.nOut
            for k in 1:length(dY[1, :]) # k < (nOut of previous layer)
                dZ[n, j] += prevW[k, j] * dY[n, k] # dY: (minibatchSize, nOut of prev. layer)
            end
            dZ[n, j] *= layer.dactivation(Z[n, j]) # Z, dZ: (minibatchSize, nOut)
            # calculate gradients of W, b
            for i in 1:layer.nIn
                gradW[j, i] += dZ[n, j] * X[n, i] # X: (minibatchSize, nIn)
            end
            gradb[j] += dZ[n, j]
        end
    end

    # update params
    for j in 1:layer.nOut
        for i in 1:layer.nIn
            # Gradient Descent method
            layer.W[j, i] -= learningRate * gradW[j, i] / minibatchSize
        end
        # Gradient Descent method
        layer.b[j] -= learningRate * gradb[j] / minibatchSize
    end

    return dZ
end