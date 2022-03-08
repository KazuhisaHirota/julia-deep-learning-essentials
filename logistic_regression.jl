include("activation_function.jl")

struct LogisticRegression
    nIn::Int # number of samples
    nOut::Int # number of features

    W::Matrix{Float64} # weights
    b::Vector{Float64} # biases
end

function LogisticRegression(nIn::Int, nOut::Int)
    w = zeros(nOut, nIn)
    b = zeros(nOut)
    return LogisticRegression(nIn, nOut, w, b)
end

# calc \sigma(Wx + b)
function output(model::LogisticRegression, x::Vector{Float64})
    # Wx + b
    preActivation = zeros(model.nOut)
    for j in 1:model.nOut
        for i in 1:model.nIn
            preActivation[j] += model.W[j, i] * x[i]
        end
        preActivation[j] += model.b[j]
    end

    return softmax(preActivation)
end

function train!(model::LogisticRegression, X::Matrix{Float64}, T::Matrix{Float64},
                minibatchSize::Int, learningRate::Float64)
    
    gradW = zeros(model.nOut, model.nIn)
    gradB = zeros(model.nOut) # nOut: number of samples

    dY = zeros(minibatchSize, model.nOut)

    # train with SGD

    # calc gradient of W, b
    for k in 1:minibatchSize
        predictedY = output(model, X[k, :])
        for j in 1:model.nOut
            dY[k, j] = predictedY[j] - T[k, j]

            for i in 1:model.nIn
                gradW[j, i] += dY[k, j] * X[k, i]
            end
            gradB[j] += dY[k, j]
        end
    end

    # update params
    for j in 1:model.nOut
        for i in 1:model.nIn
            model.W[j, i] -= learningRate * gradW[j, i] / minibatchSize
        end
        model.b[j] -= learningRate * gradB[j] / minibatchSize
    end

    return dY
end

function predict(model::LogisticRegression, x::Vector{Float64})
    y = output(model, x) # y: probability vector
    argMax = argmax(y)

    t = zeros(model.nOut) # t: label casted to 0 or 1
    for j in 1: model.nOut
        j == argMax ? t[j] = 1 : 0
    end

    return t
end