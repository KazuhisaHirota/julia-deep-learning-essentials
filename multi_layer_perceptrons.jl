include("hidden_layer.jl")
include("logistic_regression.jl")

struct MultiLayerPerceptrons
    nIn::Int
    nHidden::Int
    nOut::Int
    hiddenLayer::HiddenLayer
    logisticLayer::LogisticRegression
end

function MultiLayerPerceptrons(nIn::Int, nHidden::Int, nOut::Int, rng::MersenneTwister)
    hiddenLayer = HiddenLayer(nIn, nHidden, rng, "tanh")
    logisticLayer = LogisticRegression(nHidden, nOut)
    return MultiLayerPerceptrons(nIn, nHidden, nOut, hiddenLayer, logisticLayer)
end

function train!(model::MultiLayerPerceptrons, X::Matrix{Float64}, T::Matrix{Float64},
                minibatchSize::Int, learningRate::Float64)
    # outputs of the hidden layer (= inputs of the output layer)
    Z = zeros(minibatchSize, model.nHidden) # NOTE "self.nIn" in the original code is wrong

    # forward the hidden layer: X => Z
    for n in 1:minibatchSize
        # Z: (minibatchSize, nHidden)
        # nHidden is the size of the inputs of the output layer
        Z[n, :] = forward(model.hiddenLayer, X[n, :])
    end

    # forward & backward the output layer: Z => Y
    dY = train!(model.logisticLayer, Z, T, minibatchSize, learningRate)

    prevW = model.logisticLayer.W # for hiddenLayer, the previous layer is logisticLayer
    # backward the hidden layer (backpropagation)
    backward!(model.hiddenLayer, X, Z, dY, prevW, minibatchSize, learningRate)
end

function predict(model::MultiLayerPerceptrons, x::Vector{Float64})
    z = output(model.hiddenLayer, x)
    return predict(model.logisticLayer, z)
end